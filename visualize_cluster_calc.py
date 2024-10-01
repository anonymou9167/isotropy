from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import pdb


def cluster_cos(key_direction, y, sample_size=2000):
    if key_direction == "16":
        y_index = np.arange(y.shape[0]) % 16
    elif key_direction == "1024":
        y_index = np.arange(y.shape[0]) // 16
    else:
        raise ValueError("key_direction should be 16 or 1024")
  
    if y.shape[0] > sample_size:
        sample = np.random.choice(y.shape[0], sample_size, replace=False)
        y = y[sample]
        y_index = y_index[sample]

    sil = []
    sil_std = []
    all_labels = []
    cands = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for k in cands:
        score = []
        kmeans = KMeans(n_clusters=k).fit(y)
        labels = kmeans.labels_
        score.append(silhouette_score(y, labels, sample_size=sample_size))
        sil.append(np.mean(score))
        sil_std.append(np.std(score))
        all_labels.append(labels)
        
    if max(sil) >= 0.1:
        best_k = cands[sil.index(max(sil))]
        labels = all_labels[sil.index(max(sil))]
        std = sil_std[sil.index(max(sil))]
    else:
        best_k = 1
        labels = np.zeros(y.shape[0])
        std = 0
    print('bestk&sil&std:', best_k, max(sil), std)
    return best_k, labels, y, y_index, max(sil)

def center(y):
    y = StandardScaler(with_std=False).fit(y).transform(y) 
    return y

def run_for_a_speed_fdd(key_direction, speed=10, postfix="", noise=None, cluster_save_path='result_cluster/'):
    fdd_cluster = []
    fdd_labels = []
    fdd_y_l = []
    fdd_y_index_iekey = []
    fdd_mms = []
    for i in range(6):
        data = np.load(f"result_embedding/fdd_full_gpt2_layer{i}_speed{speed}{postfix}.npy")
        y = np.vstack(data)
        y = center(y)
        best_k, labels, y_l, y_index, mms = cluster_cos(key_direction, y, sample_size=2000)
        fdd_cluster.append(best_k)
        fdd_labels.append(labels)
        fdd_y_l.append(y_l)
        fdd_y_index_iekey.append(y_index)
        fdd_mms.append(mms)
    
    print(f"key{key_direction}, speed{speed}, noise{noise}, fdd_cluster by layers: {fdd_cluster}")

   
    with open(f"result_cluster/cluster_count_mms.txt", 'a') as f:
        f.write(f"key{key_direction}, speed{speed}, noise{noise}, fdd_cluster by layers: {fdd_cluster}, mms: {fdd_mms}\n")
    with open(f"result_cluster/cluster_count.txt", 'a') as f:
        f.write(f"key{key_direction}, speed{speed}, noise{noise}, fdd_cluster by layers: {fdd_cluster}\n")
    
    
    for i in range(6):
        np.save(f"{cluster_save_path}/dire{key_direction}_fdd_full_gpt2_layer{i}_speed{speed}{postfix}_cluster.npy", fdd_labels[i])
        np.save(f"{cluster_save_path}/dire{key_direction}_fdd_full_gpt2_layer{i}_speed{speed}{postfix}.npy", fdd_y_l[i])
        np.save(f"{cluster_save_path}/dire{key_direction}_fdd_full_gpt2_layer{i}_speed{speed}{postfix}_index.npy", fdd_y_index_iekey[i])


def run_for_a_speed_tdd(key_direction, speed=10, postfix="", noise=None, cluster_save_path='result_cluster/'):
    tdd_cluster = []
    tdd_labels = []
    tdd_y_l = []
    tdd_y_index_iekey = []
    tdd_mms = []
    for i in range(6):
        data = np.load(f"result_embedding/tdd_full_gpt2_layer{i}_speed{speed}{postfix}.npy")
        y = np.vstack(data)
        y = center(y)
        best_k, labels, y_l, y_index, mms = cluster_cos(key_direction, y, sample_size=2000)
        tdd_cluster.append(best_k)
        tdd_labels.append(labels)
        tdd_y_l.append(y_l)
        tdd_y_index_iekey.append(y_index)
        tdd_mms.append(mms)
    
    print(f"key{key_direction}, speed{speed}, noise{noise}, tdd_cluster by layers: {tdd_cluster}")
   
    with open(f"result_cluster/cluster_count_mms.txt", 'a') as f:
        f.write(f"key{key_direction}, speed{speed}, noise{noise}, tdd_cluster by layers: {tdd_cluster}, mms: {tdd_mms}\n")
    with open(f"result_cluster/cluster_count.txt", 'a') as f:
        f.write(f"key{key_direction}, speed{speed}, noise{noise}, tdd_cluster by layers: {tdd_cluster}\n")
    
    for i in range(6):
        np.save(f"{cluster_save_path}/dire{key_direction}_tdd_full_gpt2_layer{i}_speed{speed}{postfix}_cluster.npy", tdd_labels[i])
        np.save(f"{cluster_save_path}/dire{key_direction}_tdd_full_gpt2_layer{i}_speed{speed}{postfix}.npy", tdd_y_l[i])
        np.save(f"{cluster_save_path}/dire{key_direction}_tdd_full_gpt2_layer{i}_speed{speed}{postfix}_index.npy", tdd_y_index_iekey[i])


def run_for_a_speed(speed=10, noise=None, dd=None, key_direction="16", mosaic=None):
    if noise is not None:
        postfix = f"_noise{noise}"
    else:
        postfix = ""
    if mosaic is not None:
        postfix += f"_mosaic{mosaic}"
    
    cluster_save_path = 'result_cluster/'
    if not os.path.exists(cluster_save_path):
        os.makedirs(cluster_save_path)
    
    if dd is None or dd == 'fdd':
        run_for_a_speed_fdd(key_direction, speed, postfix, noise, cluster_save_path)
    if dd is None or dd == 'tdd':
        run_for_a_speed_tdd(key_direction, speed, postfix, noise, cluster_save_path)


if __name__ == '__main__':
    for speed in [100, 50, 10]:
        run_for_a_speed(speed, noise=18, dd='tdd', key_direction="16")
        run_for_a_speed(speed, noise=5, dd='tdd', key_direction="16")
        run_for_a_speed(speed, noise=18, dd='fdd', key_direction="16")
