import lightning as L
from torch.utils.data import  DataLoader
from torch.utils.data import Dataset





class DataModule(L.LightningDataModule):

    def __init__(self,
                    train_dataset: Dataset,
                    val_dataset: Dataset,
                    test_dataset: Dataset,
                    batch_size=64,
                    num_workers=4,
                    prediction_on = "test",
                    global_variable=None):
            super().__init__()
            print('-'*20)
            print(f'Global variables for data are {global_variable}')
            print('-'*20)
            
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.prediction_on = prediction_on
                 
    def setup(self, stage: str):
         pass


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def predict_dataloader(self):

        if self.prediction_on == "test":
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        elif self.prediction_on == "val":
            return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        elif self.prediction_on == "train":
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

 




    





    




