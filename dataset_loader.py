import os
import os.path
import torch
import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import json
import cv2
# TODO: Make target_field optional for unannotated datasets.
class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                 loader=default_loader, transform=None,
                 target_transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 split=None, environment=None, onlylabels=None, 
                 subset=None):
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform
        self.target_transform = target_transform
        self.add_extension = add_extension
        self.environment = environment
        self.onlylabels = onlylabels
        self.subset = subset
        self.data = pd.read_csv(csv_file)
 

        def binary_convert(x):
            if x > 0.6:
                return 1
            else:
                return 0

        # apply function to column
        if self.target_field not in ["label", "specific_label", "genus_bin"]:
            self.data[self.target_field] = self.data[self.target_field].apply(binary_convert)


        # Environments
        if self.environment is not None:
            all_envs = ["dark_corner","hair","gel_border","gel_bubble","ruler","ink","patches"]
            for env in all_envs:
                if env in self.environment:
                    self.data = self.data[self.data[env] >= 0.6]
                else:
                    self.data = self.data[self.data[env] <= 0.6]
            self.data = self.data.reset_index()


        # Only Label
        if self.onlylabels is not None:
            self.onlylabels = [int(i) for i in self.onlylabels]
            self.data = self.data[self.data[self.target_field].isin(self.onlylabels)]
            self.data = self.data.reset_index()
   
        # Subset
        if self.subset is not None:
            self.data = self.data[self.data['image'].isin(self.subset)]
            self.data = self.data.reset_index()
 
        # Split
        if split is not None:
            with open(split, 'r') as f:
                selected_images = f.read().splitlines()
            self.data = self.data[self.data[image_field].isin(selected_images)]
            self.data = self.data.reset_index()

        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(self.data[self.target_field].value_counts())
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls]
                                for cls in self.data[self.target_field]]
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]

        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()

        if type(limit) == int:
            limit = (0, limit)
        if type(limit) == tuple:
            self.data = self.data[limit[0]:limit[1]]
            self.data = self.data.reset_index()

        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        print('Found {} images from {} classes.'.format(len(self.data),
                                                        len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.data[self.target_field].value_counts())
            print("    Class '{}' ({}): {} images.".format(
                class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension:
            path = path + self.add_extension
        sample = self.loader(path)
        target = self.class_to_idx[self.data.loc[index, self.target_field]]
        
        if self.transform is not None:
            #try:
            sample = self.transform(sample)
            #except:
            #sample = cv2.imread(path)
            #sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
            #augmented = self.transform(image=sample)
            #sample = augmented['image']

            #sample = np.array(sample)
            #sample = self.transform(image=sample.astype(np.uint8))["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, target

    def __len__(self):
        return len(self.data)


class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        return super().__getitem__(i), name
    
class CSVDatasetWithMask(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        mask = Image.open('/deconstructing-bias-skin-lesion/isic2019-seg-299/{}.png'.format(name))
        mask = get_transform_mask(mask).cpu().data.numpy()
        mask = np.squeeze(mask)
        mask[mask>0.1] = 1.0
        mask[mask<=0.1] = 0.0
        return super().__getitem__(i), mask


class CSVDatasetWithMaskPH2(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        mask = Image.open('/hadatasets/abissoto/ph2images/{}_segmentation.png'.format(name))
        mask = get_transform_mask(mask).cpu().data.numpy()
        mask = np.squeeze(mask)
        mask[mask>0.1] = 1.0
        mask[mask<=0.1] = 0.0
        return super().__getitem__(i), mask

class CSVDatasetWithMaskOthers(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        mask = Image.open('/deconstructing-bias-skin-lesion/seg_outputs/resnet101_imagenet2/{}.png'.format(name))
        mask = Image.fromarray(np.array(mask)*255)
        mask = get_transform_mask(mask).cpu().data.numpy()
        mask = np.squeeze(mask)
        mask[mask>0.1] = 1.0
        mask[mask<=0.1] = 0.0
        return super().__getitem__(i), mask
 
class CSVDatasetWithNameAndMask(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        mask = Image.open('/deconstructing-bias-skin-lesion/isic2019-seg-299/{}.png'.format(name))
        mask = get_transform_mask(mask).cpu().data.numpy()
        mask = np.squeeze(mask)
        mask[mask>0.1] = 1.0
        mask[mask<=0.1] = 0.0
        return super().__getitem__(i), name, mask

class CSVDatasetWithGroups(CSVDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs_df = pd.read_csv('/group_DRO/isic_inferred_wocarcinoma.csv')
        self.attrs_df = self.attrs_df[self.attrs_df['image'].isin(self.data['image'])]
        self.confounder_names = ["dark_corner","ruler","hair","ink","patches"]
        self.attrs_df = self.attrs_df.drop(labels='image', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()
        print(self.attr_names)

        for cfd in self.confounder_names:
            self.attrs_df[cfd][self.attrs_df[cfd] > 0.6] = 1
            self.attrs_df[cfd][self.attrs_df[cfd] <= 0.6] = 0

        self.attrs_df = self.attrs_df.values
        # Get the y values
        target_idx = self.attr_idx(self.target_field)
        #print('target idx', target_idx)
        self.y_array = self.attrs_df[:, target_idx]
        #print("y:", self.y_array, np.unique(self.y_array))
        self.n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        #print('confounder_idx', self.confounder_idx)
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        #print('confounder_id', confounder_id)
        self.confounder_array = confounder_id

        # Map to groups
        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')
        
        self._group_array = torch.LongTensor(self.group_array)
        self._y_array = torch.LongTensor(self.y_array.astype(np.float))
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_field} = {int(y)}'
        # Changed from the usual behavior, because in the current experiments, we are not dealing
        # with combinations of all possible confounders, as in the skin artefacts, for example
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        group_name += ', group= {}'.format(c)
        return group_name

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        group = self.group_array[i]
        return super().__getitem__(i), group


class CSVDatasetWithKeypoints(CSVDataset):
    """
    CSVData that also returns image names.
    """
    def __init__(self, *args, max_keypoints=None, use_mask=False, **kwargs):
        super().__init__(*args, **kwargs)
        json_data = json.load(open("artifacts_keypoints.json"))
        self.use_mask = use_mask
        self.max_keypoints = max_keypoints
        # Prepare lists to store DataFrame columns
        img_names = []
        x_coords = []
        y_coords = []
        labels = []

        # Define new image dimensions
        new_width, new_height = 224, 224

        # Loop through each image in the data
        for img_data in json_data:
            # Extract image name from path
            img_name = img_data['img'].split('/')[-1].split(".")[0]

            # If image has annotations, extract them
            if 'kp-1' in img_data:
                keypoints = img_data['kp-1']
                for kp in keypoints:

                    # Calculate scaling factors
                    width_scale = new_width / 100
                    height_scale = new_height / 100

                    # Scale x and y coordinates
                    x = self.ensure_in_bounds(int(kp['x'] * width_scale))
                    y = self.ensure_in_bounds(int(kp['y'] * height_scale))
                    
                    img_names.append(img_name)
                    x_coords.append(x)
                    y_coords.append(y)
                    labels.append(kp['keypointlabels'][0])
            else:  # If no annotations, append empty values
                img_names.append(img_name)
                x_coords.append([])
                y_coords.append([])
                labels.append("empty")

        # Create a DataFrame from the data
        df = pd.DataFrame({
            'img_name': img_names,
            'x_coord': x_coords,
            'y_coord': y_coords,
            'labels': labels
        })

        # Set 'img_name' as the index
        df.set_index('img_name', inplace=True)

        original_img_names = df.index.unique()

        # Remove rows with certain labels
        df = df[~df['labels'].isin(['hair','gel_bubble','gel_border'])]

        # Add empty entries for 'img_name's that no longer exist in the DataFrame
        for img_name in original_img_names:
            if img_name not in df.index:
                df.loc[img_name] = [[], [], 'empty']
        
        if self.max_keypoints is not None:
            df = self.select_random_keypoints(df, self.max_keypoints)
            
        self.keypoints = df
        
    
    def ensure_list(self, var):
        if isinstance(var, int):
            return np.array([var])
        else:
            return np.array(var)
        
    def ensure_in_bounds(self, coord):
        coord = max(0, coord)  # Ensure coord is no less than 0
        coord = min(223, coord)  # Ensure coord is no more than 223    
        return coord 
    
    def select_random_keypoints(self, df, max_keypoints):
        """
        Randomly select a subset of keypoints for each image in the DataFrame.

        :param df: DataFrame containing the keypoints.
        :param max_keypoints: Maximum number of keypoints to keep per image.
        :return: New DataFrame with the randomly selected keypoints.
        """
        def select_keypoints(group):
            """Select keypoints from one group (corresponding to one image)."""
            num_keypoints = len(group)
            if num_keypoints <= max_keypoints:
                return group
            else:
                selected_indices = np.random.choice(num_keypoints, max_keypoints, replace=False)
                return group.iloc[selected_indices]

        df = df.reset_index()
        df = df.groupby('img_name').apply(select_keypoints).reset_index(drop=True)
        df = df.set_index('img_name')
        return df
    
    # Select from mask #
    def randomCordinates_n(self, a, n_samples, value, rng):
        # Find indices of elements in a that are equal to the target value
        sel = np.transpose(np.nonzero(a == value))
        if len(sel) >= n_samples:
            selected_idx = rng.choice(len(sel), size=n_samples, replace=False)
            x = sel[selected_idx][:,0]
            y = sel[selected_idx][:,1]
        else:
            x = [0]
            y = [0]
        return x,y
    
    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        path = os.path.join(self.root,
                            self.data.loc[i, self.image_field])
        name = self.data.loc[i, self.image_field]
        
        if self.add_extension:
            path = path + self.add_extension
        sample = self.loader(path)
        target = self.class_to_idx[self.data.loc[i, self.target_field]]
        
        mask = Image.open('/deconstructing-bias-skin-lesion/isic2019-seg-299/{}.png'.format(name)).resize((224,224))
        mask = np.array(mask) / 255.0
        mask = np.squeeze(mask)
        
        mask[mask>0.1] = 1.0
        mask[mask<=0.1] = 0.0
       
        if self.max_keypoints is not None:
            size_pad = self.max_keypoints
        else:
            size_pad = 20
        
        # get positive coords
        rng = np.random.default_rng(i)
        #rng = np.random.default_rng(1111)
        pos_coords = {}
        pos_x, pos_y = self.randomCordinates_n(mask, size_pad, 1.0, rng)
        pos_coords["x_coord"] = pos_x
        pos_coords["y_coord"] = pos_y
        
        
        # get negative coords
        if self.use_mask:
            rng = np.random.default_rng(i)
            #rng = np.random.default_rng(1111)
            coords = {}
            pos_x, pos_y = self.randomCordinates_n(mask, size_pad, 0.0, rng)
            coords["x_coord"] = pos_x
            coords["y_coord"] = pos_y
        else:
            coords = self.keypoints.loc[name].copy()
            
        # data augmentation #
        if self.transform is not None:
            sample, pos_coords, coords = self.transform((sample, pos_coords, coords))

            
        # process positive coords
        pos_x_coords = pos_coords["x_coord"]
        pos_y_coords = pos_coords["y_coord"]
        pos_coords = [(xi, yi) for xi, yi in zip(pos_x_coords, pos_y_coords)]
        pos_coords = torch.Tensor(pos_coords)
        
        # process negative coords    
        if self.use_mask:
            x = coords["x_coord"]
            y = coords["y_coord"]
        else:
            x_coords = self.ensure_list(coords["x_coord"])
            y_coords = self.ensure_list(coords["y_coord"])
            if len(x_coords) == 0:
                x = np.zeros(size_pad)
                y = np.zeros(size_pad)
            else: 
                x = np.pad(x_coords, (0, size_pad-len(x_coords)), mode="wrap")
                y = np.pad(y_coords, (0, size_pad-len(y_coords)), mode="wrap")

        coord = [(xi, yi) for xi, yi in zip(x, y)]
        coord = torch.Tensor(coord)   
        
        return sample, target, pos_coords, coord, name

    
def get_transform_mask(mask):
    
    transform_mask = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.0], [1.0])
    ])
    
    return transform_mask(mask)
    
