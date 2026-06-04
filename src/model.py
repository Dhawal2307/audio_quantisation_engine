import torch as th
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class ResidualVectorQuantizer:
    def __init__(self, n_layers=8, n_clusters=256, batch_size=2048, random_state=42):
        self.n_layers = n_layers
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        self.codebooks = None
        self.all_kmeans = []
        self.bits_per_layer = int(np.ceil(np.log2(self.n_clusters)))
        if self.bits_per_layer * self.n_layers > 64:
            raise ValueError(
                f"Configuration limit reached: Total bits requested ({self.bits_per_layer * self.n_layers}) "
                f"exceeds single 64-bit allocation limits. Reduce layers or cluster sizes."
            )
        

    def fit(self, training_data_scaled: np.ndarray, device: th.device):
        current_residuals = training_data_scaled.copy()             ## size of training data -> [t, n_mels]
        list_cb = []
        for layer_idx in range(self.n_layers):
            print(f"Training Layer {layer_idx + 1}/{self.n_layers}...")
            km = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=self.batch_size, random_state=self.random_state)
            km.fit(current_residuals)                           ## this fits and finds the k clusters from the residuals
            cb = th.tensor(km.cluster_centers_ , dtype=th.float32)
            list_cb.append(cb)
            self.all_kmeans.append(km)
            
            indices = km.predict(current_residuals)
            reconstructed_mel = km.cluster_centers_[indices]
            current_residuals = current_residuals - reconstructed_mel
        self.codebooks = th.stack(list_cb).to(device)
        print("All RVQ layers trained successfully.")
    
    def get_indices(self, mel_scaled: np.ndarray) -> np.ndarray:
        '''
        Function to get the indices for any audio_mel based on the trained codebooks of all n_layers
        Shape of mel_scaled -> [t,n_mels]
        '''
        indices_matrix = []
        current_residual = mel_scaled.copy()                

        for km in self.all_kmeans:
            predicted_indices = km.predict(current_residual)
            indices_matrix.append(predicted_indices)
            current_residual = current_residual - km.cluster_centers_[predicted_indices]
        return np.array(indices_matrix, dtype= np.int32)        ## shape here will be [n_layer, t]
    
    def pack_indices(self,indices_matrix: np.ndarray) -> np.ndarray:
        n_layers , n_frames = indices_matrix.shape
        packed_data = np.zeros(n_frames, dtype = np.int64)
        
        for i in range(n_layers):
            layer = indices_matrix[i].astype(np.int64)
            packed_data |= (layer << (i*self.bits_per_layer))
        return packed_data