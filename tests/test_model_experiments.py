import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import cv2
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_limits

from src.feature_experiments import extract_study, file_hash, load_study, prepare_study
from src.final_calibration import finalize
from src.knn import KNNAnomalyDetector
from src.model_experiments import (
    REPRESENTATIONS, choose_provisional, compare_models, load_development,
    make_representation, parameter_grid, score_summary,
)
from src.ocsvm import CalibratedOneClassSVM


class Stage4DetectorTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)
        self.rng = np.random.default_rng(42)
        self.fit = self.rng.normal(size=(50, 4))
        self.calibration = self.rng.normal(size=(40, 4))

    def test_svm_historico_e_gamma_calculado_apenas_no_fit(self):
        detector = CalibratedOneClassSVM().fit(self.fit)
        historical = OneClassSVM(kernel="rbf", nu=.05, gamma="scale").fit(self.fit)
        np.testing.assert_array_equal(detector.score_samples(self.calibration), historical.score_samples(self.calibration))
        self.assertEqual(detector.resolved_gamma_, historical._gamma)
        scaled = CalibratedOneClassSVM(gamma_multiplier=2).fit(self.fit)
        self.assertEqual(scaled.resolved_gamma_, 2 * historical._gamma)
        before = scaled.resolved_gamma_
        scaled.calibrate(self.calibration * 100)
        self.assertEqual(before, scaled.resolved_gamma_)

    def test_scores_limiar_predicoes_e_ids_para_ambos_modelos(self):
        for detector in (CalibratedOneClassSVM(), KNNAnomalyDetector(), KNNAnomalyDetector(aggregation="kth")):
            detector.fit(self.fit, sample_ids=[f'f{i}' for i in range(50)])
            with self.assertRaises(NotFittedError):
                detector.predict(self.calibration)
            with self.assertRaisesRegex(ValueError, "Vazamento"):
                detector.calibrate(self.calibration, sample_ids=[f'f{i}' for i in range(40)])
            detector.calibrate(self.calibration, sample_ids=[f'c{i}' for i in range(40)])
            score = detector.anomaly_score(self.calibration)
            self.assertEqual(detector.threshold_, np.quantile(score, .95, method="higher"))
            np.testing.assert_array_equal(detector.score_samples(self.calibration), -score)
            np.testing.assert_array_equal(detector.predict(self.calibration), np.where(score > detector.threshold_, -1, 1))
            np.testing.assert_allclose(detector.decision_function(self.calibration), detector.threshold_ - score)
            self.assertLessEqual(detector.calibration_fpr_, .05)
            detector.fit(self.fit)
            with self.assertRaises(NotFittedError):
                detector.predict(self.calibration)

    def test_kth_sem_autovizinhanca_e_media_preservada(self):
        X = np.array([[0.], [1.], [3.], [8.], [12.]])
        distances = np.abs(X - X.T)
        np.fill_diagonal(distances, np.inf)
        nearest = np.sort(distances, axis=1)[:, :3]
        for aggregation in ("mean", "kth"):
            detector = KNNAnomalyDetector(3, aggregation=aggregation).fit(X)
            expected = nearest.mean(axis=1) if aggregation == 'mean' else nearest[:, -1]
            np.testing.assert_array_equal(detector.training_scores_, expected)
            self.assertFalse(np.any(detector.training_neighbor_indices_ == np.arange(5)[:, None]))
        with self.assertRaises(ValueError):
            KNNAnomalyDetector(aggregation='median').fit(X)

    def test_invalidos_clone_reproducao_e_serializacao(self):
        for kwargs in ({'nu':0}, {'nu':float('nan')}, {'gamma_multiplier':0}, {'gamma_multiplier':float('inf')}):
            with self.assertRaises(ValueError):
                CalibratedOneClassSVM(**kwargs).fit(self.fit)
        for detector in (CalibratedOneClassSVM(), KNNAnomalyDetector(aggregation='kth')):
            first = detector.fit(self.fit).calibrate(self.calibration)
            second = clone(detector).fit(self.fit).calibrate(self.calibration)
            np.testing.assert_array_equal(first.predict(self.calibration), second.predict(self.calibration))
            buffer = io.BytesIO()
            joblib.dump(first, buffer)
            buffer.seek(0)
            np.testing.assert_array_equal(first.score_samples(self.calibration), joblib.load(buffer).score_samples(self.calibration))
        with self.assertRaises(ValueError):
            CalibratedOneClassSVM().fit(self.fit, y=-np.ones(50))
        with self.assertRaises(ValueError):
            score_summary([1., float('nan')])


class Stage4ProtocolTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root/'data'
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        warn = warnings.catch_warnings()
        warn.__enter__()
        self.addCleanup(warn.__exit__, None, None, None)
        warnings.filterwarnings('ignore', message='Calibração pequena:')
        for split, count, offset in (('train',80,0), ('test/good',4,150)):
            folder = self.data/split
            folder.mkdir(parents=True)
            for i in range(count):
                _, encoded = cv2.imencode('.png', np.full((8,8,3), i+offset, dtype=np.uint8))
                (folder/f'{i:03}.png').write_bytes(encoded.tobytes())
        self.study = self.root/'study'
        self.cache = self.root/'cache'
        prepare_study(self.data, self.study, train_flat_is_good=True)
        with patch('src.feature_experiments.processar_imagem', side_effect=self.fake_extract):
            extract_study(self.study, self.cache, configs=['A','C'], full=True)
        self.small_grid = [
            {'model':'svm','parameters':{'nu':.05,'gamma_multiplier':1.0}},
            {'model':'knn','parameters':{'n_neighbors':3,'aggregation':'mean'}}]

    def fake_extract(self, path, bank, *, hog_config):
        return np.random.default_rng(int(path.stem)).normal(size=hog_config.total_dimension).astype(np.float32), True

    def compare(self, out):
        with patch('src.model_experiments.parameter_grid', return_value=self.small_grid):
            return compare_models(self.study, self.cache, out)

    def test_grade_pequena_e_representacoes_corretas(self):
        self.assertEqual(len(parameter_grid()),17)
        for name, dim in (('R1',25413),('R2',43),('R3',25)):
            self.assertEqual(REPRESENTATIONS[name]['dimension'],dim)
            transform=make_representation(name)
            self.assertEqual(transform.hog_dimension,25380 if name=='R1' else 3864)

    def test_busca_reproduzivel_sem_acesso_a_reservas_e_sem_vazamento(self):
        _,outer,_=load_study(self.study)
        forbidden=set(outer['calibration_ids']) | set(outer['test_ids'])
        def guarded_hash(path):
            self.assertNotIn(str(path),forbidden)
            return file_hash(path)
        with patch('src.model_experiments.file_hash',side_effect=guarded_hash):
            first=self.compare(self.root/'one')
            second=self.compare(self.root/'two')
        self.assertEqual(len(first['rows']),18)
        for a,b in zip(first['rows'],second['rows']):
            self.assertEqual(a['threshold'],b['threshold'])
            self.assertEqual(a['validation_scores'],b['validation_scores'])
            self.assertFalse(set(a['calibration_ids']) & set(a['validation_ids']))
            self.assertFalse((set(a['calibration_ids']) | set(a['validation_ids'])) & forbidden)
            self.assertEqual(a['n_fit'],40)
        for model in ('svm','knn'):
            a=choose_provisional(first['summary'])[model]
            b=choose_provisional(second['summary'])[model]
            self.assertEqual(a['config_key'],b['config_key'])

    def test_manifesto_cache_ordem_e_hash_corrompidos_sao_rejeitados(self):
        path=self.study/'internal_manifest.json'
        original=path.read_text(); manifest=json.loads(original)
        manifest['folds'][0]['fit_ids'][0]=manifest['folds'][0]['validation_ids'][0]
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):load_development(self.study,self.cache)
        path.write_text(original)
        cache=self.cache/'features_A.npz'
        with np.load(cache) as a:
            arrays={k:a[k][::-1] for k in a.files}
        with cache.open('wb') as f:np.savez_compressed(f,**arrays)
        with self.assertRaisesRegex(ValueError,'Hash'):load_development(self.study,self.cache)
        meta_path=self.cache/'features_A.json';meta=json.loads(meta_path.read_text())
        meta['cache_sha256']=file_hash(cache);meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'Ordem'):load_development(self.study,self.cache)

    def test_piloto_nao_seleciona_e_nunca_sobrescreve(self):
        out=self.root/'pilot'
        report=compare_models(self.study,self.cache,out,pilot=True)
        self.assertEqual(len(report['rows']),2)
        self.assertFalse((out/'selection.json').exists())
        with self.assertRaises(ValueError):choose_provisional(report['summary'])
        before=file_hash(out/'comparison.json')
        with self.assertRaises(FileExistsError):compare_models(self.study,self.cache,out)
        self.assertEqual(before,file_hash(out/'comparison.json'))

    def test_transformadores_ajustados_apenas_no_fit(self):
        fits=[]
        from src.representations import FeatureRepresentation
        original=FeatureRepresentation.fit
        def tracked(obj,X,y=None):
            fits.append(len(X))
            return original(obj,X,y)
        with patch.object(FeatureRepresentation,'fit',tracked):
            self.compare(self.root/'tracked')
        self.assertEqual(fits,[40]*9)

    def test_calibracao_final_separada_somente_em_fixture_sintetico(self):
        selection=self.root/'selected'
        self.compare(selection)
        with self.assertRaisesRegex(ValueError,'confirm-final'):
            finalize(self.study,self.cache,selection/'selection.json',self.root/'blocked',model_name='svm')
        _,outer,_=load_study(self.study)
        with patch('src.final_calibration.processar_imagem',side_effect=self.fake_extract) as extract:
            out=finalize(self.study,self.cache,selection/'selection.json',self.root/'final',model_name='svm',confirm=True)
        self.assertEqual([str(call.args[0]) for call in extract.call_args_list],outer['calibration_ids'])
        bundle=joblib.load(out/'model.joblib')
        self.assertEqual(bundle['representation'].scaler_.n_samples_seen_,64)
        self.assertEqual(set(bundle['detector'].fit_ids_),set(outer['fit_ids']))
        self.assertEqual(set(bundle['detector'].calibration_ids_),set(outer['calibration_ids']))
        self.assertFalse(set(bundle['detector'].calibration_ids_) & set(outer['test_ids']))
        with self.assertRaises(FileExistsError):
            finalize(self.study,self.cache,selection/'selection.json',out,model_name='svm',confirm=True)


if __name__ == '__main__':
    unittest.main()
