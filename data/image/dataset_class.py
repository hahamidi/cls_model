import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class AdvancedCLS(Dataset):
    def __init__(
        self,
        csv_path,
        image_root_path,
        image_size=320,
        class_index=-1,
        use_frontal=True,
        use_upsampling=False,
        flip_label=False,
        shuffle=False,
        seed=123,
        verbose=True,
        upsampling_cols=None,
        train_cols=None,
        mode='train',
        uncertain_label_map=None,
        classifier_free_training_probability=0.0,
        ratio = 1.0
    ):
        # Set default values for optional parameters
        if upsampling_cols is None:
            upsampling_cols = []
        if train_cols is None:
            train_cols = [
                'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',
                'Pleural Effusion', 'No Finding'
            ]
        if uncertain_label_map is None:
            uncertain_label_map = {
                'Cardiomegaly': 0,
                'Edema': 1,
                'Consolidation': 0,
                'Atelectasis': 1,
                'Pleural Effusion': 0,
                'No Finding': 0
            }

        self.image_size = image_size
        self.mode = mode
        self.class_index = class_index
        self.classifier_free_training_probability = classifier_free_training_probability

        # Load data from CSV
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
        if ratio < 1.0:
            self.df = self.df.sample(frac=ratio, random_state=seed).reset_index(drop=True)
            if not shuffle:
                print("************Warning: ratio is less than 1.0 but shuffle is set to False. This may lead potential problems.************")

        # Map race and gender categories if included in training columns
        for col, mapper in [('race_group', self.map_race_category), ('gender', self.map_gender_category)]:
            if col in train_cols:
                self.df[col] = self.df[col].apply(mapper)

        # Normalize age if included
        if 'anchor_age' in train_cols:
            self.df['anchor_age'] /= 100.0

        # Filter to frontal images if specified
        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']

        # Upsample specified columns
        if use_upsampling and upsampling_cols:
            sampled_dfs = [self.df[self.df[col] == 1] for col in upsampling_cols]
            self.df = pd.concat([self.df] + sampled_dfs, axis=0)

        # Handle uncertain labels and fill missing values
        for col in train_cols:
            if col in uncertain_label_map:
                self.df[col] = self.df[col].replace(-1, uncertain_label_map[col])
            self.df[col] = self.df[col].fillna(0)

        # Optionally flip labels
        if flip_label and class_index != -1:
            self.df[train_cols] = self.df[train_cols].replace(0, -1)

        # Shuffle data
        if shuffle:
            self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)

        # Prepare image and label lists
        assert image_root_path, 'Please provide the correct dataset path!'
        self._images_list = (image_root_path + self.df['Path']).tolist()
        if class_index != -1:
            self._labels_list = self.df[train_cols].iloc[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()

        self._num_images = len(self.df)

        if verbose:
            for col in train_cols:
                print(self.df[col].value_counts(dropna=False))

    @staticmethod
    def map_race_category(race):
        if isinstance(race, str):
            race = race.upper()
            if 'ASIAN' in race:
                return 0
            elif 'BLACK' in race or 'AFRICAN AMERICAN' in race:
                return 1
            elif 'WHITE' in race:
                return 2
            elif 'HISPANIC' in race or 'LATINO' in race:
                return 3
        return 3  # Other or unknown

    @staticmethod
    def map_gender_category(gender):
        if gender == 'M':
            return 0
        elif gender == 'F':
            return 1
        else:
            return np.random.choice([0, 1])

    def image_augmentation(self, image):
        img_aug = transforms.Compose([
            transforms.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=128
            ),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
        return img_aug(image)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        # Load image in grayscale mode and apply augmentations
        image = Image.open(self._images_list[idx]).convert('L')
        if self.mode == 'train':
            image = self.image_augmentation(image)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size))

        # Transform image to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
        ])
        image = transform(image)

        # Prepare label
        label = np.array(self._labels_list[idx], dtype=np.float32)
        sample = {'data': image, 'labels': label}
        return sample

if __name__ == "__main__":
    data = MIMICAdvancedCLS(
        csv_path="/home/hhamidi/clean/per_storage/csvs/mimic/mimic_test_fairness.csv",
        image_root_path="/datasets/mimic/mimic-cxr-jpg/2.0.0/",
        class_index= -1,
    )
    print(data[0])
