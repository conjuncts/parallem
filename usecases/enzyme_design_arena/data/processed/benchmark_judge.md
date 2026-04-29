### nucb_functional_cls
Ranking:
1. svc_rbf — test_accuracy 0.8279569892473119 (test_f1 0.6666666666666666)  
2. gb_cls — test_accuracy 0.8279569892473119 (test_f1 0.6363636363636364)  
3. svc_linear — test_accuracy 0.8172043010752689 (test_f1 0.6222222222222222)

Selected model: svc_rbf  
Justification: svc_rbf ties for highest test accuracy and has the best test F1 among the top scorers, indicating superior overall test-set classification performance.

---

### nucb_num_mut_reg
Ranking:
1. svr_rbf — test_rmse 1.3952536025267026 (test_r2 0.8649581303613914)  
2. gb_reg — test_rmse 1.7647356526447557 (test_r2 0.7839663235413654)  
3. rf_reg — test_rmse 2.0623960395688403 (test_r2 0.704942764469379)

Selected model: svr_rbf  
Justification: svr_rbf achieves the lowest test RMSE and the highest test R2, clearly outperforming others on the test set for this regression task.

---

### proteingym_total_mut_reg
Ranking:
1. rf_reg — test_rmse 63478.936150317284 (test_r2 0.017480345548728016)  
2. elasticnet — test_rmse 63784.316155907334 (test_r2 0.00800433394704625)  
3. ridge_high — test_rmse 64024.67741025257 (test_r2 0.0005138850737550493)

Selected model: rf_reg  
Justification: rf_reg has the lowest test RMSE (and the highest test R2) among evaluated models on the test set, so it is the best performer here.