# Pontos de operação da etapa 9

Análise complementar e exploratória. O holdout já foi consultado nas avaliações anteriores. As cinco configurações foram reajustadas sobre as mesmas 256 imagens boas com ajuste cruzado. Os limiares foram definidos apenas com a calibração ou os defeitos de seleção. As 20 boas da metade de seleção aparecem em scores.csv para registro.

A decisão usa score > limiar. Para os alvos de seleção, usa-se o maior float64 abaixo do score de corte, incluindo empates. O limiar original é o quantil 95% superior das boas reservadas. A AUC usa os scores do holdout e é constante entre limiares de cada modelo.

Os scores de calibração e seleção foram persistidos aqui pela primeira vez. A verificação compara o limiar original, as métricas de seleção, os scores e predições do holdout e as métricas finais.
