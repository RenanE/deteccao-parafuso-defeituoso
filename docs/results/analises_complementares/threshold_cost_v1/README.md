# Otimização exploratória de limiares

O holdout já havia sido consultado. Os pesos foram fixados em 1, 2, 3, 5 e 10. O ajuste de J = w × FNR + FPR usou apenas 58 defeitos de seleção e 64 boas de calibração. Nenhuma escolha de peso ou limiar usa o holdout.

A decisão é score > limiar. Os candidatos são os scores distintos do ajuste de limiar e o float64 anterior ao mínimo, que permite rejeitar todos. Em empate de custo, vence a menor FNR e, se persistir, o maior limiar candidato. O máximo observado representa a decisão de aprovar todos no ajuste de limiar.

comparison.csv reporta as taxas e os custos nos cinco pesos para os três pontos anteriores e para os cinco novos por modelo. AUC é calculada dos scores do holdout e independe do limiar. pareto.csv inclui pontos não dominados por modelo e indica os também não dominados globalmente entre os 40 pontos avaliados. Empates de FN e FP preservam todas as estratégias equivalentes.
