import numpy as np 
import pandas as pd
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from keras import backend as K

from sklearn.metrics.pairwise import cosine_similarity

LATENT_DEST_EMBEDDING_DIM = 149
NUM_CLUSTERS = 100

DRIVE = "D:/kiki/Documents/"
DATA_FOLDER = DRIVE+'data/'
CSV_EXTENSION = '.csv'
PICKLE_DIR= DRIVE+'pckls/'
MODEL_FOLDER = DRIVE+'models/'

SRCH_DEST_ID = 'srch_destination_id'
USER_ID = 'user_id'
HOTEL_CLUSTER = 'hotel_cluster'



def pickle_it(to_pickle, fname):
    with open(PICKLE_DIR+fname+'.pkl', 'wb') as f:
        pickle.dump(to_pickle, f)
    print('done')

def destinations_preprocessing(): #should actually be in other file
    print('in destinations_preprocessing')
    destinations = pd.read_csv(DATA_FOLDER+'destinations'+CSV_EXTENSION)
    dst_id_to_idx_map = {k: v for v, k in destinations.srch_destination_id.items()}
    destinations = destinations.drop(columns=[SRCH_DEST_ID]).to_numpy()
    average_destination = destinations.mean(axis=0)
    pickle_it(destinations, 'destinations_inp_numpy')
    pickle_it(average_destination, 'avg_destinations_np')
    pickle_it(dst_id_to_idx_map, 'dst_id_to_idx_map')

    return destinations

def get_inputs(ds_set):
    print('reading inputs for', ds_set)
    read_pickles= []
    fname_lst = [
        'hotel_df_in_csv_lst', 'hotel_df_not_in_csv_lst', 'user_df_not_in_csv_lst', 
        'user_df_in_csv_lst', 'hotel_df', 'avg_destinations_np', 'user_dest_id_mapping',
        'dst_id_to_idx_map', 'candidate_generation_label', 'users_df', 'user_id_to_embedding_idx_mapping', 'overall_avg_user',
        'search_df', 'overall_avg_search'
    ]
    not_split_pkls = set(['avg_destinations_np', 'dst_id_to_idx_map'])
    diff_names = {
        'avg_destinations_np': 'avg_destination_embedding', 
        'user_df_not_in_csv_lst': 'user_df_1',
        'user_df_in_csv_lst': 'user_df_2',
        'hotel_dst_embedded_vec_id_in_csv': 'hotel_dst_1',
        'hotel_dst_embedded_vec_id_NOT_in_csv': 'hotel_dst_2'
    }
    read_pickles = {}
    for fname in fname_lst:
        path = PICKLE_DIR+fname+ds_set+'.pkl' if fname not in not_split_pkls else PICKLE_DIR+fname+'.pkl'
        k = fname if fname not in diff_names else diff_names[fname]
        read_pickles[k] = pd.read_pickle(path)
        # read_pickles.append(pd.read_pickle(path))
    
    return read_pickles
    
    # hotel_dst_embedded_vec_id_in_csv, hotel_dst_embedded_vec_id_NOT_in_csv, user_df_not_in_csv_lst, user_df_in_csv_lst, hotel_df, avg_destinations_np, user_dest_id_mapping, dst_id_to_idx_map, candidate_generation_label, users_df, user_id_to_embedding_idx_mapping, overall_avg_user, search_df, overall_avg_search = read_pickles # make this into a dict
    # inputs = {
    #     'user_df_1': user_df_not_in_csv_lst,
    #     'user_df_2': user_df_in_csv_lst,
    #     'hotel_dst_1': hotel_dst_embedded_vec_id_in_csv,
    #     'hotel_dst_2': hotel_dst_embedded_vec_id_NOT_in_csv,
    #     'hotel_df': hotel_df,
    #     'avg_destination_embedding': avg_destinations_np,
    #     'user_dest_id_mapping': user_dest_id_mapping,
    #     'dst_id_to_idx_map': dst_id_to_idx_map,
    #     'candidate_generation_label': candidate_generation_label,
    #     'users_df': users_df,
    #     'user_id_to_embedding_idx_mapping': user_id_to_embedding_idx_mapping, 
    #     'overall_avg_user': overall_avg_user,
    #     'search_df': search_df,
    #     'overall_avg_search': overall_avg_search
    # }

    # return inputs

def _create_destination_embeddings_helper(destination_embeddings_param, sorted_user_ids, user_dest_id_mapping, dst_id_to_idx_map, avg_destination_embedding):
    print('creating destination embedding features')
    # destination_embeddings_param = tf.make_ndarray(destination_embeddings_param)
    destination_embeddings = []
    for i in sorted_user_ids:
        to_repeat = 0
        dest_ids = user_dest_id_mapping[i]
        lst_of_idxs = []
        for j in dest_ids:
            if j in dst_id_to_idx_map:
                lst_of_idxs.append(dst_id_to_idx_map[j])
            else:
                to_repeat += 1
        to_add = destination_embeddings_param[lst_of_idxs, :]
        
        # replacing the would-be destination embeddings w/ avg embeddings for those ids not in csv
        avgs = pd.DataFrame(np.tile(avg_destination_embedding, (to_repeat, 1)))
        destination_embeddings.append(np.mean(np.concatenate((to_add, avgs), axis=0)), axis=0)

    destination_embeddings = np.array(destination_embeddings)
    return destination_embeddings

def candidate_generation_feature_generation(destination_embeddings_param, other_embeddings):
    print('generating candidate features')
    hotel_embedding_preprocessed = other_embeddings['hotel_df']
    # user_embedding_1 = other_embeddings['user_df_1']
    # user_embedding_2 = other_embeddings['user_df_2']
    user_embedding = other_embeddings['users_df']
    avg_destination_embedding = other_embeddings['avg_destination_embedding']
    user_dest_id_mapping = other_embeddings['user_dest_id_mapping']
    dst_id_to_idx_map = other_embeddings['dst_id_to_idx_map']
    # candidate_generation_label = other_embeddings['candidate_generation_label']

    # can probably be moved to the other file
    hotel_embeddings_pivot = hotel_embedding_preprocessed.pivot_table(index=USER_ID) # in asc order; pivot_table handles averaging already
    hotel_embeddings_np = hotel_embeddings_pivot.to_numpy()
    user_embeddings_np = user_embedding.sort_values(by=USER_ID).to_numpy() # should match order
    destination_embeddings_np = _create_destination_embeddings_helper(destination_embeddings_param, hotel_embeddings_pivot.index, user_dest_id_mapping, dst_id_to_idx_map, avg_destination_embedding) # should match hotel order

    return hotel_embeddings_np, user_embeddings_np, destination_embeddings_np

def hotel_cluster_embedding_feature_generation(other_embeddings):
    # hotel_destination embedding pivoted by hotel cluster
    hotel_dest_embed_1 = other_embeddings['hotel_dst_1']
    hotel_dest_embed_2 = other_embeddings['hotel_dst_2']
    hotel_dest_np = pd.concat([hotel_dest_embed_1, hotel_dest_embed_2]).pivot_table(index=HOTEL_CLUSTER).to_numpy()
    
    # user_embedding pivoted by hotel cluster
    user_embed_cluster_lst_1 = other_embeddings['user_df_1']
    user_embed_cluster_lst_2 = other_embeddings['user_df_2']
    user_embed_cluster_np = pd.concat(user_embed_cluster_lst_1 + user_embed_cluster_lst_2).pivot_table(index=HOTEL_CLUSTER).to_numpy()

    return hotel_dest_np, user_embed_cluster_np

def ranking_feaure_generation(hotel_cluster_embeddings_param, other_embeddings,  generated_hotel_clusters_lst):
    ## search feature (don't forget to drop hotel_cluster after)
    search_df = other_embeddings['search_df']
    user_ids = search_df.user_id

    ## cosine similarity features
    user_id_to_embedding_idx_mapping = other_embeddings['user_id_to_embedding_idx_mapping']
    overall_avg_user = other_embeddings['overall_avg_user']
    overall_avg_search = other_embeddings['overall_avg_search']    

    pass
    


def make_embedding_model(inp_shape, embedding_layer_out_dim=100):
    inp = keras.Input(shape=(inp_shape,)) # know from ds that each destination is made up of 149 latent var
    embedding_layer = layers.Dense(embedding_layer_out_dim) # output layer is 100 bc 100 hotel clusters
    embeddings = dest_embedding_layer(inp)
    embeddings_model = Model(inputs=inp, outputs=embeddings)
    return embeddings_model


def main():
    # destinations = destinations_preprocessing()

    train_embedings_dict = get_inputs('_train')
    val_embedings_dict = get_inputs('_val')
    train_candidate_generation_label = train_embedings_dict['candidate_generation_label'].to_numpy()
    val_candidate_generation_label = val_embedings_dict['candidate_generation_label'].to_numpy
    destinations = pd.read_pickle(PICKLE_DIR+'destinations_inp_numpy.pkl')

    ## destination vector embedding model (could add more layers to make more complex if want)
    # model
    destination_embeddings_model = make_embedding_model(149)
    # dest_inputs = keras.Input(shape=(149,)) # know from ds that each destination is made up of 149 latent var
    # dest_embedding_layer = layers.Dense(100) # output layer is 100 bc 100 hotel clusters
    # destination_embeddings = dest_embedding_layer(dest_inputs)
    # destination_embeddings_model = Model(inputs=dest_inputs, outputs=destination_embeddings)

    idx = 1 # TODO: check if index out model layer for embeddings really is 1
    destination_embedding_output = destination_embeddings_model.layers[idx].output 
    ## or may need to do something like passing embeddigs_model.outputs -- but don't know

    ## candidate generation model
    # model preparation
    # destinations is the same for both train and val since it's only used for learning embeddings
    hotel_embeddings, user_embeddings, destination_embeddings = candidate_generation_feature_generation(destination_embedding_output, train_embedings_dict)
    hotel_embeddings_np_val, user_embeddings_np_val, destination_embeddings_np_val = candidate_generation_feature_generation(destination_embedding_output, val_embedings_dict)

    hotel_inputs = layers.Input(shape=(4,))
    user_inputs = layers.Input(shape=(6,))
    destination_inputs = layers.Input(shape=(100,))
    inp = layers.Inputs(shape=(110,))
    # avg_layer = layers.Average()
    concat_layer = layers.Concatenate(axis=2)
    dense_layer_256 = layers.Dense(256, activation='relu')
    dense_layer_512 = layers.Dense(512, activation='relu')
    final_layer = layers.Dense(100, activation='softmax')

    # model
    candidate_generation_input = concat_layer([destination_inputs, hotel_inputs, user_inputs])
    x = dense_layer_256(candidate_generation_input)
    x = dense_layer_512(x)
    x = dense_layer_256(x)
    candidate_generation_out = final_layer(x)
    candidate_generation_model_intermediary = Model(inputs=[candidate_generation_input] , outputs=candidate_generation_out)

    candidate_generation_model = Model(
        inputs=[destination_embeddings_model.input, candidate_generation_model_intermediary.input],
        outputs=candidate_generation_out
    )

    ## hotel cluster embedding model
    # model preparation
    hotel_dest_np, user_embed_cluster_np = hotel_cluster_embedding_feature_generation(train_embedings_dict)
    hotel_dest_np_val, user_embed_cluster_np_val = hotel_cluster_embedding_feature_generation(val_embedings_dict)
    # model
    hotel_cluster_embeddings_model = make_embedding_model(3+149+5) # inp shape = hotel_dest(hotel+destiation) + user_embed_cluster
    # hotel_cluster_inps = keras.Input(shape=(3+149+5,)) 
    # hotel_cluster_embedding_layer = layers.Dense(100) # output layer is 100 bc 100 hotel clusters
    # hotel_cluster_embeddings = hotel_cluster_embedding_layer(hotel_cluster_inps)
    # hotel_cluster_embeddings_model = Model(inputs=hotel_cluster_inps, outputs=hotel_cluster_embedding_layer)

    ## ranking model




    # maybe compile should be for entire model
    candidate_generation_model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )



    print(candidate_generation_model.summary())

    candidate_generation_model.fit(
        x=[destinations, hotel_embeddings, user_embeddings, destination_embeddings],
        y=train_candidate_generation_label, 
        vaidation_data=([destinations, hotel_embeddings_np_val, user_embeddings_np_val, destination_embeddings_np_val], val_candidate_generation_label),
        epochs=100, batch_size=32
    )

    candidate_generation_model.save(MODEL_FOLDER+'candidate_generation_model')



    ## ranking model
    # model preparation



if __name__ == "__main__":
    main()





    