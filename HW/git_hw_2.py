################### Criteria ###################
def processSubset(self, X,y,feature_index):
    # Fit model on feature_set and calculate rsq_adj
    regr = sm.OLS(y,X[:,feature_index]).fit()
    rsq_adj = regr.rsquared_adj
    bic = self.myBic(X.shape[0], regr.mse_resid, len(feature_index))
    rsq = regr.rsquared
    return {"model":regr, "rsq_adj":rsq_adj, "bic":bic, "rsq":rsq, "predictors_index":feature_index}

################### Forward Stepwise ###################
def forward(self,predictors_index,X,y):
    # Pull out predictors we still need to process
    remaining_predictors_index = [p for p in range(X.shape[1])
                            if p not in predictors_index]

    results = []
    for p in remaining_predictors_index:
        new_predictors_index = predictors_index+[p]
        new_predictors_index.sort()
        results.append(self.processSubset(X,y,new_predictors_index))
        # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    # Choose the model with the highest rsq_adj
    # best_model = models.loc[models['bic'].idxmin()]
    best_model = models.loc[models['rsq'].idxmax()]
    # Return the best model, along with model's other  information
    return best_model

def forwardK(self,X_est,y_est, fK):
    models_fwd = pd.DataFrame(columns=["model", "rsq_adj", "bic", "rsq", "predictors_index"])
    predictors_index = []

    M = min(fK,X_est.shape[1])

    for i in range(1,M+1):
        print(i)
        models_fwd.loc[i] = self.forward(predictors_index,X_est,y_est)
        predictors_index = models_fwd.loc[i,'predictors_index']

    print(models_fwd)
    # best_model_fwd = models_fwd.loc[models_fwd['bic'].idxmin(),'model']
    best_model_fwd = models_fwd.loc[models_fwd['rsq'].idxmax(),'model']
    # best_predictors = models_fwd.loc[models_fwd['bic'].idxmin(),'predictors_index']
    best_predictors = models_fwd.loc[models_fwd['rsq'].idxmax(),'predictors_index']
    return best_model_fwd, best_predictors

