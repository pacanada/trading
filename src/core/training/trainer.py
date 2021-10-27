from src.core.training.wandblogger import WandbLogger
import pandas as pd
import copy
class Trainer:
    def __init__(self,
                 model,
                 training_data:pd.DataFrame,
                 test_data:pd.DataFrame,
                 validation_data:pd.DataFrame,
                wandb_logger: WandbLogger):
        self.model = copy.deepcopy(model)
        self.training_data= training_data
        self.test_data=test_data
        self.validation_data=validation_data
        self.score_training = []
        self.score_test = []
        self.score_validation = []
        self.predictions_info = []
        self.wandb_logger = wandb_logger

    def get_training_info_dict(self, it):
        training_info_dict = {"epochs": it,
                     "score_training":self.score_training[-1:][0],
                    "score_test": self.score_test[-1:][0],
                    "score_validation": self.score_validation[-1:][0],
                    **self.predictions_info[-1:][0],
                    }
        return training_info_dict

    def init_training(self, epochs, features, target, save_every=10, print_every=10, name_artifact="NeuralNetwork"):
        threshold_val =-0.1
        
        for it in range(epochs):
            self.model.fit(self.training_data[features], self.training_data[target].values.ravel())
            if it%save_every==0:
                self.score_training.append(self.model.score(self.training_data[features],self.training_data[target]))
                self.score_test.append(self.model.score(self.test_data[features],self.test_data[target]))
                self.score_validation.append(self.model.score(self.validation_data[features],self.validation_data[target]))
                preds = self.model.predict(self.validation_data[features])
                self.predictions_info.append({"pred_mean": preds.mean(), "pred_max": preds.max(), "pred_min":preds.min()})
                training_info_dict = self.get_training_info_dict(it=it)
                self.wandb_logger.log(training_info_dict)
            if it%print_every==0:
                training_info_dict = self.get_training_info_dict(it=it)
                print(training_info_dict)

            if (self.score_validation[-1:][0] > threshold_val) | (epochs==(it+1)):
                training_info_dict = self.get_training_info_dict(it=it)
                self.wandb_logger.log(training_info_dict)
                print("Saved model")
                self.wandb_logger.log_artifact(model=copy.deepcopy(self.model), name=name_artifact, metrics_dict=training_info_dict)
                self.best_model = copy.deepcopy(self.model)
                threshold_val = self.score_validation[-1:][0]