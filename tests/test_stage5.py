import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_limits

import test_model_experiments as stage4_tests
from src.feature_experiments import file_hash, load_study, signature
from src.final_calibration import finalize
from src.isolation_forest import IsolationForestAnomalyDetector
from src.mahalanobis import MahalanobisAnomalyDetector, NumericalFitError
from src.model_experiments import compare_models, model_array_bytes, parameter_grid
from src.selection_bundle import prepare_bundle


class NewDetectorTests(unittest.TestCase):
    def setUp(self):
        limits = threadpool_limits(limits=1)
        self.addCleanup(limits.restore_original_limits)
        rng = np.random.default_rng(42)
        self.X = rng.normal(size=(120, 4))
        self.cal = rng.normal(size=(40, 4))

    def models(self):
        return (MahalanobisAnomalyDetector(), MahalanobisAnomalyDetector(covariance="elliptic"),
                IsolationForestAnomalyDetector(n_estimators=15, max_samples=64))

    def test_ledoit_wolf_distancias_batem_com_solucao_linear(self):
        detector = MahalanobisAnomalyDetector().fit(self.X)
        expected = LedoitWolf().fit(self.X)
        diff = self.cal - expected.location_
        distances = np.sum(diff * np.linalg.solve(expected.covariance_, diff.T).T, axis=1)
        np.testing.assert_allclose(detector.anomaly_score(self.cal), distances)
        np.testing.assert_array_equal(detector.score_samples(self.cal), -detector.anomaly_score(self.cal))
        self.assertGreater(detector.diagnostics_["shrinkage"], 0)

    def test_covariancia_singular_regularizada_e_robusta_rejeitada(self):
        X = np.tile(np.arange(40)[:, None], (1, 4)).astype(float)
        model = MahalanobisAnomalyDetector().fit(X)
        self.assertTrue(np.isfinite(model.anomaly_score(X)).all())
        self.assertEqual(model.diagnostics_["input_rank"], 1)
        with self.assertRaises(NumericalFitError):
            MahalanobisAnomalyDetector(covariance="elliptic").fit(X)
        with self.assertRaises(NumericalFitError):
            MahalanobisAnomalyDetector(covariance="elliptic").fit(self.X[:8])
        with self.assertRaises(NumericalFitError):
            MahalanobisAnomalyDetector().fit(np.ones((30,4)))

    def test_covariancia_alta_dimensao_e_condicionamento_rejeitados(self):
        with self.assertRaisesRegex(ValueError,"256 atributos"):
            MahalanobisAnomalyDetector().fit(np.ones((3,257)))
        bad = self.X * np.array([1e10,1,1,1])
        with self.assertRaises(NumericalFitError):
            MahalanobisAnomalyDetector(covariance="elliptic").fit(bad)
        with self.assertRaises(NumericalFitError):
            MahalanobisAnomalyDetector().fit(self.X * 1e200)

    def test_contrato_calibracao_scores_predicoes_sem_vazamento(self):
        for model in self.models():
            with self.subTest(model=type(model).__name__):
                model.fit(self.X, sample_ids=[f'f{i}' for i in range(120)])
                with self.assertRaises(NotFittedError):model.predict(self.cal)
                with self.assertRaisesRegex(ValueError,'Vazamento'):
                    model.calibrate(self.cal, sample_ids=[f'f{i}' for i in range(40)])
                model.calibrate(self.cal, sample_ids=[f'c{i}' for i in range(40)])
                scores = model.anomaly_score(self.cal)
                self.assertEqual(model.threshold_,np.quantile(scores,.95,method='higher'))
                np.testing.assert_array_equal(model.score_samples(self.cal),-scores)
                np.testing.assert_allclose(model.decision_function(self.cal),model.threshold_-scores)
                np.testing.assert_array_equal(model.predict(self.cal),np.where(scores>model.threshold_,-1,1))
                self.assertTrue(np.all(model.predict(np.full((3,4),100))==-1))
                model.fit(self.X)
                with self.assertRaises(NotFittedError):model.predict(self.cal)

    def test_clone_seed_serializacao_e_calibracao_nao_refaz_fit(self):
        for model in self.models():
            first = model.fit(self.X).calibrate(self.cal)
            second = clone(model).fit(self.X).calibrate(self.cal)
            np.testing.assert_array_equal(first.score_samples(self.cal),second.score_samples(self.cal))
            before = first.score_samples(self.X).copy()
            first.calibrate(self.cal * 3)
            np.testing.assert_array_equal(before,first.score_samples(self.X))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.joblib'
                joblib.dump(first,path)
                np.testing.assert_array_equal(first.predict(self.cal),joblib.load(path).predict(self.cal))

    def test_if_offset_nativo_ignorado_e_memoria_das_arvores(self):
        model=IsolationForestAnomalyDetector(n_estimators=10,max_samples=64).fit(self.X)
        self.assertEqual(model.model_.contamination,'auto')
        np.testing.assert_array_equal(model.score_samples(self.cal),model.model_.score_samples(self.cal))
        model.calibrate(self.cal)
        expected=model.predict(self.cal)
        model.model_.offset_=999.
        np.testing.assert_array_equal(expected,model.predict(self.cal))
        tree_bytes=sum(sum(a.nbytes for a in t.tree_.__getstate__().values() if isinstance(a,np.ndarray)) for t in model.model_.estimators_)
        self.assertGreaterEqual(model_array_bytes(model),tree_bytes)

    def test_entradas_invalidas_e_parametros_sem_clipping(self):
        for kwargs in ({'n_estimators':0},{'max_samples':121},{'max_samples':True},
                       {'max_features':0},{'random_state':None}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                IsolationForestAnomalyDetector(**kwargs).fit(self.X)
        with self.assertRaises(ValueError):IsolationForestAnomalyDetector(max_samples=16).fit(np.ones((30,4)))
        for model in self.models():
            with self.assertRaises(ValueError):model.fit(self.X,y=-np.ones(120))
            with self.assertRaises(ValueError):model.fit(np.full((120,4),np.nan))
            model.fit(self.X)
            with self.assertRaises(ValueError):model.anomaly_score(np.zeros((3,5)))
            with self.assertRaises(ValueError):model.calibrate(self.cal,y=-np.ones(40))


class Stage5ProtocolTests(unittest.TestCase):
    setUp = stage4_tests.Stage4ProtocolTests.setUp
    fake_extract = stage4_tests.Stage4ProtocolTests.fake_extract
    compare4 = stage4_tests.Stage4ProtocolTests.compare

    def compare5(self,out):
        grid=[{'model':'mahalanobis','parameters':{'covariance':c}} for c in ('ledoit_wolf','elliptic')]
        grid += [{'model':'isolation_forest','parameters':{'n_estimators':5,'max_samples':16,'max_features':f,'random_state':42}} for f in (.5,1.)]
        with patch('src.model_experiments.parameter_grid',return_value=grid):
            return compare_models(self.study,self.cache,out,stage=5)

    def test_grade_controlada_e_random_state_fixo(self):
        grid=parameter_grid(5)
        self.assertEqual(len(grid),11)
        self.assertEqual(sum(s['model']=='isolation_forest' for s in grid),8)
        self.assertTrue(all(s['parameters']['random_state']==42 for s in grid))

    def test_comparacao_reproduzivel_com_inviaveis_sem_reservas(self):
        _,outer,_=load_study(self.study)
        forbidden=set(outer['calibration_ids'])|set(outer['test_ids'])
        def guarded(path):
            self.assertNotIn(str(path),forbidden)
            return file_hash(path)
        with patch('src.model_experiments.file_hash',side_effect=guarded):
            first=self.compare5(self.root/'first')
            second=self.compare5(self.root/'second')
        self.assertEqual(len(first['rows']),24)
        self.assertEqual(len(first['skipped']),6)
        self.assertTrue(all(r['parameters']['covariance']=='elliptic' for r in first['skipped']))
        self.assertFalse(any(r['representation']=='R1' and r['model']=='mahalanobis' for r in first['rows']+first['skipped']))
        for a,b in zip(first['rows'],second['rows']):
            self.assertEqual(a['validation_scores'],b['validation_scores'])
            self.assertEqual(a['threshold'],b['threshold'])
            self.assertTrue('numerical_diagnostics' in a)
        selection=json.loads((self.root/'first/selection.json').read_text(encoding="utf-8"))
        self.assertEqual(selection['selected']['mahalanobis']['parameters']['covariance'],'ledoit_wolf')
        with self.assertRaises(FileExistsError):self.compare5(self.root/'first')

    def test_bundle_preserva_escolhas_e_bloqueia_originais_alterados(self):
        old,new=self.root/'stage4',self.root/'stage5'
        self.compare4(old);self.compare5(new)
        hashes={str(p):file_hash(p) for folder in (old,new,self.study) for p in folder.iterdir() if p.is_file()}
        merged=prepare_bundle(self.study,self.cache,old/'selection.json',new/'selection.json',self.root/'merged')
        selection=json.loads((merged/'selection.json').read_text(encoding="utf-8"))
        self.assertEqual(set(selection['selected']),{'svm','knn','mahalanobis','isolation_forest'})
        self.assertEqual(selection['selected']['svm'],json.loads((old/'selection.json').read_text(encoding="utf-8"))['selected']['svm'])
        self.assertTrue(all(file_hash(p)==h for p,h in hashes.items()))
        for model in ('mahalanobis','isolation_forest'):
            with self.assertRaisesRegex(ValueError,'confirm-final'):
                finalize(self.study,self.cache,merged/'selection.json',self.root/f'blocked{model}',model_name=model)
            with patch('src.final_calibration.processar_imagem',side_effect=self.fake_extract) as extractor:
                final=finalize(self.study,self.cache,merged/'selection.json',self.root/f'final{model}',model_name=model,confirm=True)
            _,outer,_=load_study(self.study)
            self.assertEqual([str(c.args[0]) for c in extractor.call_args_list],outer['calibration_ids'])
            bundle=joblib.load(final/'model.joblib')
            self.assertEqual(bundle['representation'].scaler_.n_samples_seen_,64)
        (old/'comparison.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'histórico'):
            finalize(self.study,self.cache,merged/'selection.json',self.root/'bad',model_name='svm',confirm=True)

    def test_migracao_nao_aceita_mudanca_metodologica(self):
        old,new=self.root/'old',self.root/'new'
        self.compare4(old);self.compare5(new)
        p=old/'selection.json';s=json.loads(p.read_text(encoding="utf-8"))
        s['sources']['knn.py']='0'*64
        s['signature']=signature({k:v for k,v in s.items() if k!='signature'})
        p.write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError,'metodológico'):
            prepare_bundle(self.study,self.cache,p,new/'selection.json',self.root/'bad')


if __name__ == '__main__':
    unittest.main()
