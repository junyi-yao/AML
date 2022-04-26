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
IDX_OF_EMBEDDINGS_OUTPUT = 0
NUM_CLUSTERS_TO_INVESTIGATE = 10

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

def get_inputs(ds_set):
    print('reading inputs for', ds_set)
    read_pickles= []
    fname_lst = [
        'hotel_df_in_csv_lst', 'hotel_df_not_in_csv_lst', 'user_df_not_in_csv_lst', 
        'user_df_in_csv_lst', 'hotel_df', 'avg_destinations_np', 'user_dest_id_mapping',
        'dst_id_to_idx_map', 'candidate_generation_label', 'users_df', 'user_id_to_embedding_idx_mapping', 'overall_avg_user',
        'search_df', 'overall_avg_search', 'label'
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
    
    return read_pickles

def _create_destination_embeddings_helper(destination_embeddings_param, sorted_user_ids, user_dest_id_mapping, dst_id_to_idx_map, avg_destination_embedding):
    print('creating destination embedding features')
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
    user_embedding = other_embeddings['users_df']
    avg_destination_embedding = other_embeddings['avg_destination_embedding']
    user_dest_id_mapping = other_embeddings['user_dest_id_mapping']
    dst_id_to_idx_map = other_embeddings['dst_id_to_idx_map']

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

    hotel_cluster_embedding_inp_np = np.concatenate([hotel_dest_np, user_embed_cluster_np],axis=1)

    return hotel_dest_np, user_embed_cluster_np, hotel_cluster_embedding_inp_np

def ranking_feaure_generation(hotel_cluster_embeddings_param, other_embeddings,  generated_hotel_clusters_lst, user_embeddings):
    ## search feature (don't forget to drop hotel_cluster after)
    search_df = other_embeddings['search_df']
    search_df_cluster = search_df.pivot_table(index=HOTEL_CLUSTER).to_numpy()
    user_ids = search_df.user_id
    hotel_clusters = search_df.hotel_cluster
    search_embedding_np = search_df_og.drop(columns=[HOTEL_CLUSTER, USER_ID]).to_numpy()

    og_num_vecs, og_size_embed = search_embedding_np.shape 
    search_df = np.repeat(search_embedding_np, num_proposed_clusters, axis=0)
    search_df = np.reshape((og_num_vecs, NUM_CLUSTERS_TO_INVESTIGATE, og_size_embed))

    ## hotel cluster and cosine feature
    user_id_to_embedding_idx_mapping = other_embeddings['user_id_to_embedding_idx_mapping']
    users_by_cluster_embeddings_np = other_embeddings['users_df'].to_numpy()

    all_hotel_cluster_embeddings = []
    all_cosine_user_fts = []    
    all_cosine_search_fts = []
    for user_id in user_ids:
        idx = user_id_to_embedding_idx_mapping[user_id] # inp into candidate generation NN is same order as user_embedding
        user_embedding = user_embeddings[idx]
        search_embedding = search_embedding_np[int(search_df.loc[search_df[USER_ID] == user_id].index.values[0])]
        
        hotel_clusters = generated_hotel_clusters_lst[idx]
        hotel_cluster_embeddings = [] # per user
        for hotel_clusters_idx in generated_hotel_clusters_lst:
            hotel_cluster_embeddings.append(hotel_cluster_embeddings_param[hotel_clusters_idx])
            avg_user_embed_for_cluster = users_by_cluster_embeddings_np[hotel_clusters_idx]
            avg_search_embed_for_cluster = search_df_cluster[hotel_clusters_idx]
            
            all_cosine_user_fts.append(cosine_similarity(user_embedding, avg_user_embed_for_cluster))
            all_cosine_search_fts.append(cosine_similarity(search_embedding, avg_search_embed_for_cluster))

        
        all_hotel_cluster_embeddings.append(hotel_cluster_embeddings)
    all_hotel_cluster_embeddings = np.array(all_hotel_cluster_embeddings)
    s = len(all_cosine_search_fts)
    all_cosine_user_fts = np.array(all_cosine_user_fts).reshape(s, 1)
    all_cosine_search_fts = np.array(all_cosine_search_fts).reshape(s,1)

    return search_embedding_np, all_hotel_cluster_embeddings_np, all_cosine_user_fts_np, all_cosine_search_fts_np


def return_final_preds(model, generated_hotel_clusters_lst):
    out = model_output.output
    top_5 = np.argsort(-out, axis=1)[:, :5]
    final_preds = []
    for i, preds in enumerate(final_preds):
        actual_hotel_cluster_pred = []
        for idx in preds:
            actual_hotel_cluster_pred.append(generated_hotel_clusters_lst[i][idx])
        final_preds.append(actual_hotel_cluster_pred)
    
    return final_preds

def main():
    train_embedings_dict = get_inputs('_train')
    val_embedings_dict = get_inputs('_val')
    train_label = train_embedings_dict['label'].to_numpy()
    val_label = val_embedings_dict['label'].to_numpy
    destinations = pd.read_pickle(PICKLE_DIR+'destinations_inp_numpy.pkl')

    ### candidate generation model
    destination_embeddings_inputs = layers.Input(shape=(149,))
    hotel_inputs = layers.Input(shape=(4,))
    user_inputs = layers.Input(shape=(6,))
    destination_inputs = layers.Input(shape=(100,))
    inp = layers.Inputs(shape=(110,))
    embedding_layer = layers.Dense(100) # output layer is 100 bc 100 hotel clusters
    concat_layer = layers.Concatenate(axis=2)
    dense_layer_256 = layers.Dense(256, activation='relu')
    dense_layer_512 = layers.Dense(512, activation='relu')
    final_layer = layers.Dense(100, activation='softmax')

    ## model
    dest_embeddings = embedding_layer(destination_embeddings_inputs)
    candidate_generation_input = concat_layer([destination_inputs, hotel_inputs, user_inputs])
    x = dense_layer_256(candidate_generation_input)
    x = dense_layer_512(x)
    x = dense_layer_256(x)
    candidate_generation_out = final_layer(x)

    candidate_generation_model = Model(
        inputs=[destination_embeddings_inputs.input, candidate_generation_input],
        outputs=[dest_embeddings, candidate_generation_out]
    )

    ## model fit preparation
    # destinations is the same for both train and val since it's only used for learning embeddings
    destination_embedding_output = candidate_generation_model.outputs[IDX_OF_EMBEDDINGS_OUTPUT]
    hotel_embeddings, user_embeddings, destination_embeddings = candidate_generation_feature_generation(destination_embedding_output, train_embedings_dict)
    hotel_embeddings_np_val, user_embeddings_np_val, destination_embeddings_np_val = candidate_generation_feature_generation(destination_embedding_output, val_embedings_dict)
    generated_hotel_clusters_lst = np.argsort(-candidate_generation_out, axis=1)[:, :NUM_CLUSTERS_TO_INVESTIGATE]



    ### hotel cluster embedding model
    hotel_cluster_embedding_inp = layers.Input(shape=(3+149+5,)) # inp shape = hotel_dest(hotel+destiation) + user_embed_cluster
    hotel_cluster_inputs = layers.Input(shape=(100,))
    search_feature_inputs = layers.Input(shape=(12,))
    cosine_similarity_inputs = layers.Input(shape=(1,))
    inp = layers.Inputs(shape=(114,))
    concat_layer = layers.Concatenate(axis=2) #TODO: check, is this right?
    final_layer_ranking = layers.Dense(10, activation='softmax')

    ## model
    hotel_cluster_embeddings = embedding_layer(hotel_cluster_embedding_inp)
    ranking_input = concat_layer([hotel_cluster_inputs, search_feature_inputs, cosine_similarity_inputs, cosine_similarity_inputs])
    x = dense_layer_256(ranking_input)
    x = dense_layer_512(x)
    x = dense_layer_256(x)
    ranking_input_out = final_layer_ranking(x)

    ranking_input_model = Model(
        inputs=[hotel_cluster_embedding_inp, hotel_cluster_inputs, search_feature_inputs, cosine_similarity_inputs, cosine_similarity_inputs],
        outputs=[hotel_cluster_embeddings, ranking_input_out]
    )

    ## model fit preparation
    hotel_cluster_embedding_output = ranking_input_model.outputs[IDX_OF_EMBEDDINGS_OUTPUT]
    hotel_dest_np, user_embed_cluster_np, hotel_cluster_embedding_inp_np = hotel_cluster_embedding_feature_generation(train_embedings_dict)
    hotel_dest_np_val, user_embed_cluster_np_val, hotel_cluster_embedding_inp_np_val = hotel_cluster_embedding_feature_generation(val_embedings_dict)
    search_embedding_np, all_hotel_cluster_embeddings_np, all_cosine_user_fts_np, all_cosine_search_fts_np = ranking_feaure_generation(hotel_cluster_embedding_output, train_embedings_dict, generated_hotel_clusters_lst, user_embed_cluster_np)
    search_embedding_np_val, all_hotel_cluster_embeddings_np_val, all_cosine_user_fts_np_val, all_cosine_search_fts_np_val = ranking_feaure_generation(hotel_cluster_embedding_output, val_embedings_dict, generated_hotel_clusters_lst, user_embed_cluster_np)

    model = Model(
        inputs=[candidate_generation_model.inputs, ranking_input_model.inputs],
        outputs=ranking_input_model.out
    )

    # maybe compile should be for entire model
    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )


    print('candidate generation model')
    print(candidate_generation_model.summary())

    print()
    print('ranking input model')
    print(ranking_input_model.summary())

    print()
    print('entire model')
    print(model.summary())


    candidate_generation_fit_x_train = [destinations, hotel_embeddings, user_embeddings, destination_embeddings]
    candidate_generation_fit_x_val = [destinations, hotel_embeddings_np_val, user_embeddings_np_val, destination_embeddings_np_val]
    ranking_fit_x_train = [hotel_cluster_embedding_inp_np, search_embedding_np, all_hotel_cluster_embeddings_np, all_cosine_user_fts_np, all_cosine_search_fts_np]
    ranking_fit_x_val = [hotel_cluster_embedding_inp_np_val, all_hotel_cluster_embeddings_np_val, all_cosine_user_fts_np_val, all_cosine_search_fts_np_val]
    
    model.fit(
        x=candidate_generation_fit_x_train + candidate_generation_fit_x_val + ranking_fit_x_train + ranking_fit_x_val,
        y=train_label, 
        vaidation_data=(candidate_generation_fit_x_val + ranking_fit_x_val, val_label),
        epochs=100, batch_size=32
    )

    candidate_generation_model.save(MODEL_FOLDER+'candidate_generation_model')
    ranking_input_model.save(MODEL_FOLDER+'ranking_input_model')
    model.save(MODEL_FOLDER+'recommender_model')



if __name__ == "__main__":
    main()





    