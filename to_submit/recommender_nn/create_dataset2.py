import numpy as np 
import pandas as pd
import pickle

DRIVE = "D:/kiki/Documents/"
DATA_FOLDER = DRIVE+'data/'
CSV_EXTENSION = '.csv'
SRCH_DEST_ID = 'srch_destination_id'
PICKLE_DIR= DRIVE+'pckls/'

LATENT_DEST_EMBEDDING_DIM = 149

USER_ID = 'user_id'
HOTEL_CLUSTER = 'hotel_cluster'
USER_FEATURES = {
    'user_id',
    'site_name',
    'posa_continent',
    'user_location_country',
    'user_location_region',
    'user_location_city'
}
HOTEL_FEATURES = {
    'hotel_continent',
    'hotel_country',
    'hotel_market',
    'hotel_cluster'
}
SEARCH_FEATURES = {
    'user_id',
    'date_time',
    'is_mobile',
    'is_package',
    'channel',
    'srch_ci',
    'srch_co',
    'srch_adults_cnt',
    'srch_children_cnt',
    'srch_rm_cnt',
    'srch_destination_type_id',
    'cnt',
    'orig_destination_distance'
}
LABELS = {
    'is_booking',
    'hotel_cluster'
}
# is_booking not important since we removed any non-booked rows

def pickle_it(to_pickle, fname):
    with open(PICKLE_DIR+fname+'.pkl', 'wb') as f:
        pickle.dump(to_pickle, f)


def make_initial_train_val_split():
    train = pd.read_csv(DATA_FOLDER+'train'+CSV_EXTENSION)

    users_who_booked = train.loc[train['is_booking'] == 1].user_id.unique()
    train_df = train.loc[train['user_id'].isin(users_who_booked)]

    # separate out users who only booked once to who booked multiple times
    users_book_only_once_ids_tmp = train_df.loc[train_df['is_booking'] == 1].user_id.value_counts() == 1
    users_book_only_once_ids = users_book_only_once_ids_tmp[users_book_only_once_ids_tmp].index
    users_book_multiple_times_ids = users_book_only_once_ids_tmp[~users_book_only_once_ids_tmp].index #  return type is an np arr

    train_df_multiple_books = train_df.loc[train_df['user_id'].isin(users_book_multiple_times_ids)]
    train_df_one_bok = train_df.loc[train_df['user_id'].isin(users_book_only_once_ids)]

    # 80/20 ish test split
    train_num = int(train_df.shape[0] * .8)
    val_num = train_df.shape[0] - train_num

    # have 80/20 split for one booking only for logic defined above
    one_booking_size = int(train_df_one_bok.shape[0] * 0.8) # have 80% of the one booking user be in the training set
    many_booking_size = train_num - one_booking_size # remainder of training set will be filled by many users

    train_df_one_book_idxs = np.random.choice(train_df_one_bok.index, size=one_booking_size, replace=False)
    val_df_one_book_idxs = np.setdiff1d(train_df_one_bok.index, train_df_one_book_idxs)

    # fill in remainder from users with multiple booking
    train_df_many_books_idxs = np.random.choice(train_df_multiple_books.index, size=many_booking_size, replace=False)
    val_df_many_books_idxs = np.setdiff1d(train_df_multiple_books.index, train_df_many_books_idxs)


    # make train/val dfs intermediary step
    train_users_book_only_once_df = train_df_one_bok.loc[list(train_df_one_book_idxs)]
    val_users_book_only_once_df = train_df_one_bok.loc[list(val_df_one_book_idxs)]

    train_user_multiple_books_df = train_df_multiple_books.loc[list(train_df_many_books_idxs)]
    val_user_multiple_books_df = train_df_multiple_books.loc[list(val_df_many_books_idxs)]

    # make train/val dfs
    print('foo')
    train_df = pd.concat([train_users_book_only_once_df, train_user_multiple_books_df])
    val_df = pd.concat([val_users_book_only_once_df, val_user_multiple_books_df])

    # shuffle 
    print('shuffle')
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = val_df.sample(frac=1).reset_index(drop=True)

    # check
    print('train')
    print(train_df.shape)
    print(train_num)

    print('val')
    print(val_df.shape)
    print(val_num)

    # save intermediary as pickle
    train_df.to_pickle(PICKLE_DIR+'init_train_df_split')
    val_df.to_pickle(PICKLE_DIR+'init_val_df_split')

    return train_df, val_df

def _select_specific_cols(df, cols):
    return df[cols]

def _create_pivot_table(df, index):
    return df.pivot_table(index=index)

def _handle_id_not_in_destinations(df, destinations):
    avg_destination_latent_var = destinations.iloc[:,1:].mean().to_numpy()  #  to use when srch_destination_id not in destinations_csv; start @ 1 bc 0 is srch_destination_id
    avg_destination_latent_var = avg_destination_latent_var.reshape(1, avg_destination_latent_var.shape[0])
    base_df_not_in_destination_csv_ids = np.setdiff1d(df.srch_destination_id.unique(), destinations.srch_destination_id.unique())
    base_df_not_in_destination_csv = df.loc[df['srch_destination_id'].isin(base_df_not_in_destination_csv_ids)]

    avg_latent_var_repeated_np = np.tile(avg_destination_latent_var, (base_df_not_in_destination_csv.shape[0], 1))
    avg_latent_var_repeated = pd.DataFrame(avg_latent_var_repeated_np)
    avg_latent_var_repeated.columns = ['d'+str(i) for i in range(1, 150)]

    base_df_not_in_destination_csv.reset_index(drop=True, inplace=True)
    avg_latent_var_repeated.reset_index(drop=True, inplace=True)

    hotel_df_for_destinations_id_not_in_csv = pd.concat([base_df_not_in_destination_csv, avg_latent_var_repeated], axis=1)
    return hotel_df_for_destinations_id_not_in_csv

def _order_hotel_by_hotel_cluster(dfs, embedded_df):
    res = []
    for dff in dfs:
        hotel_clusters_in_dff = dff.index
        embedded_attrs = embedded_df[embedded_df.index.isin(hotel_clusters_in_dff)] # already sorted by nature since hotel cluster is the index
        res.append(embedded_attrs)
    return res

def _order_user_by_hotel_cluster(dfs, embedded_df):
    res = []
    for dff in dfs:
        hotel_clusters_in_dff = dff.index
        embedded_attrs = embedded_df[embedded_df.hotel_cluster.isin(hotel_clusters_in_dff)].sort_values(by='hotel_cluster')
        res.append(embedded_attrs)
    return res

def _order_search_by_hotel_cluster(embedded_df):
    return embedded_df.sort_values(by='hotel_clusters')


def order_attr_by_hotel_cluster(dfs, embedded_df, embedded_df_type):
    if embedded_df_type == 'hotel':
        return _order_hotel_by_hotel_cluster(dfs, embedded_df)
    elif embedded_df_type == 'user':
        return _order_user_by_hotel_cluster(dfs, embedded_df)
    elif embedded_df_type == 'search':
        return _order_search_by_hotel_cluster(embedded_df)

def _handle_id_in_destinations(destinations):
    # TODO: FIXME
    dst_id_to_idx_map = {k: v for v, k in destinations.srch_destination_id.items()}
    inp = destinations.drop(columns=['srch_destination_id'])
    embedding_model = keras.Sequential()
    embedding_model.add(layers.Embedding(LATENT_DEST_EMBEDDING_DIM, 100))
    embedding_model.compile

def make_destination_embedded_vecs(df, ds_discrim):
    ## destination srch vec
    # id not in destinations csv
    destinations = pd.read_csv(DATA_FOLDER+'destinations'+CSV_EXTENSION)
    hotel_df_for_destinations = _select_specific_cols(df, ['srch_destination_id', 'hotel_cluster'])
    hotel_df_for_destinations_id_not_in_csv = _handle_id_not_in_destinations(hotel_df_for_destinations, destinations)
    avg_latent_for_hotel_cluster_not_in_csv_lst = [hotel_df_for_destinations_id_not_in_csv.groupby('hotel_cluster').mean()] # already ordered by hotel cluster asc
    print('destination embed id_not_in_csv done')

    # id in destinations csv
    # order by srch_dst_id so can join in keras layer
    srch_d_id_in_df = hotel_df_for_destinations.srch_destination_id.isin(destinations.srch_destination_id)
    hotel_df_for_destinations_id_in_csv = hotel_df_for_destinations[srch_d_id_in_df] # has hotel_cluster in feature
    all_destination_embed = destinations[destinations.srch_destination_id.isin(srch_d_id_in_df)]
    hotel_df_for_destinations_partitions = np.array_split(hotel_df_for_destinations_id_in_csv, 3)
    avg_latent_for_hotel_cluster_in_csv_lst = [ df.merge(all_destination_embed, on='srch_destination_id', how='inner').groupby('hotel_cluster').mean() for df in hotel_df_for_destinations_partitions]
    print('destination embed id_in_csv done')

    for dff in (avg_latent_for_hotel_cluster_in_csv_lst + avg_latent_for_hotel_cluster_not_in_csv_lst):
        dff.drop(columns=['srch_destination_id'])
    print('avg_latent_for_hotel_clusters done')

    pickle_it(avg_latent_for_hotel_cluster_not_in_csv_lst, 'avg_latent_for_hotel_cluster_not_in_csv_lst'+ds_discrim)
    pickle_it(avg_latent_for_hotel_cluster_in_csv_lst, 'avg_latent_for_hotel_cluster_in_csv_lst'+ds_discrim)

    return avg_latent_for_hotel_cluster_not_in_csv_lst, avg_latent_for_hotel_cluster_in_csv_lst

def _combine_hotel_and_dest_embeddings(hotel_embedding_lst, dest_embedding_lst):
    hotel_dst_to_union = []
    for idx, hotel_dff in enumerate(hotel_embedding_lst):
        hotel_dst_to_union.append(dest_embedding_lst[idx].merge(hotel_dff, on='hotel_cluster', how='inner'))
    hotel_dst_embed = pd.concat(hotel_dst_to_union)
    return hotel_dst_embed 


def make_candidate_generation_datasets(df, ds_discrim):
    print('candidate generation')
    ## user vec (index is user_id; _ x 6)
    users_df = _select_specific_cols(df, list(USER_FEATURES) + ['hotel_cluster'])
    users_df = _create_pivot_table(users_df, index='user_id')
    print('users done')
    
    ## destination srch vec (index is hotel cluster; 99x150)
    dest_embed_id_not_in_dest_csv_lst, dest_embed_id_in_csv_lst = make_destination_embedded_vecs(df, ds_discrim)

    ## hotel vec (index is hotel_cluster, 99x3)
    hotel_df = _select_specific_cols(df, list(HOTEL_FEATURES)).groupby('hotel_cluster').mean() #  fine that the ints get averaged into a float; since everything would be avg anyway
    hotel_df.to_pickle(PICKLE_DIR+'hotel_df'+ds_discrim+'.pkl')
    
    # list of hotel_dfs where df at element i has same hotel_clusters as the ones in dest_embed_id_in_csv_lst
    hotel_df_in_csv_lst = order_attr_by_hotel_cluster(dest_embed_id_in_csv_lst, hotel_df, 'hotel')
    hotel_df_not_in_csv_lst = order_attr_by_hotel_cluster(dest_embed_id_not_in_dest_csv_lst, hotel_df, 'hotel')

    ## trying to join now for sake of ease
    hotel_dst_embedded_vec_id_in_csv = _combine_hotel_and_dest_embeddings(hotel_df_in_csv_lst, dest_embed_id_in_csv_lst)
    hotel_dst_embedded_vec_id_NOT_in_csv = _combine_hotel_and_dest_embeddings(hotel_df_not_in_csv_lst, dest_embed_id_not_in_dest_csv_lst)
    pickle_it(hotel_df_in_csv_lst, 'hotel_df_in_csv_lst'+ds_discrim)
    pickle_it(hotel_df_not_in_csv_lst, 'hotel_df_not_in_csv_lst'+ds_discrim)
    pickle_it(hotel_dst_embedded_vec_id_in_csv, 'hotel_dst_embedded_vec_id_in_csv'+ds_discrim)
    pickle_it(hotel_dst_embedded_vec_id_NOT_in_csv, 'hotel_dst_embedded_vec_id_NOT_in_csv'+ds_discrim)


    # ordering user attrs by hotel cluster to easily map to hotel attrs so can do concat in keras layer (just do union in keras)
    user_df_not_in_csv_lst = order_attr_by_hotel_cluster(dest_embed_id_not_in_dest_csv_lst, users_df, 'user')
    user_df_in_csv_lst = order_attr_by_hotel_cluster(dest_embed_id_in_csv_lst, users_df, 'user')
    print('hotel_df done')

    pickle_it(user_df_not_in_csv_lst, 'user_df_not_in_csv_lst'+ds_discrim)
    pickle_it(user_df_in_csv_lst, 'user_df_in_csv_lst'+ds_discrim)

    return hotel_dst_embedded_vec_id_in_csv, hotel_dst_embedded_vec_id_NOT_in_csv, user_df_not_in_csv_lst, user_df_in_csv_lst

# def make_test_candidate_generation_dataset(df):
#     destinations = pd.read_csv(DATA_FOLDER+'destinations'+CSV_EXTENSION)

#     to_select = USER_FEATURES.union(HOTEL_FEATURES)
#     to_select.remove('hotel_cluster')

#     user_hotel_embeddings = _select_specific_cols(df, list(to_select)+['srch_destination_id']) # base df is already user and hotel embeddings
    
#     dst_embed_id_NOT_in_csv_df = _handle_id_not_in_destinations(user_hotel_embeddings, destinations)
#     dst_embed_in_csv_df = user_hotel_embeddings.merge(destinations, on='srch_desination_id', how='inner')
#     base_df = pd.concat([base_df1, base_df2])

#     base_df.pickle(PICKLE_DIR+'test_generation_inp'+'_test')

#     return base_df


def _check_if_pickle(df):
    res = df
    if type(df) == str:
        res = pd.read_pickle(df)
    return res
        

def make_ranking_datasets(df, user_df_1, user_df_2, hotel_dst_1, hotel_dst_2, ds_discrim):
    search_df = _select_specific_cols(df, list(SEARCH_FEATURES) + ['srch_destination_id']) # already in the order it should be

    user_df_1 = _check_if_pickle(user_df_1)
    user_df_2 = _check_if_pickle(user_df_2)
    user_df = pd.concat(user_df_1 + user_df_2)

    hotel_dst_1 = _check_if_pickle(hotel_dst_1)
    hotel_dst_2 = _check_if_pickle(hotel_dst_2)
    hotel_dst_df = pd.concat(hotel_dst_1 + hotel_dst_2)

    ranking_df = search_df.join(user_df, on=['user_id'], how='left').join(hotel_dst_df, on=['srch_destination_id'], how='left')
    label = ranking_df[['hotel_cluster']]
    ranking_df.drop(columns=['user_id', 'hotel_cluster']) #  removing answer from final ranking ds; also don't need user_id

    ranking_df.to_pickle(PICKLE_DIR+'ranking_inp_df'+ds_discrim+'.pkl')  

    return ranking_df, label   

def make_user_dest_id_mapping(df, ds_discrim):
    user_dest_id_mapping = {}
    a = df[[SRCH_DEST_ID, USER_ID]]
    user_ids = a.user_id.unique()
    tot_num = user_ids.shape[0]
    cnt = 0
    print(tot_num)
    for i in user_ids:
        if cnt%5000 == 0:
            print(cnt)
        user_dest_id_mapping[i] = a.loc[a[USER_ID] == i].srch_destination_id.values
        cnt += 1
    print('dict created')
    pickle_it(user_dest_id_mapping, 'user_dest_id_mapping'+ds_discrim)
    return user_dest_id_mapping

def make_labels(df, ds_discrim):
    a = _select_specific_cols(df, [USER_ID, 'hotel_cluster'])
    one_hot = pd.get_dummies(a[HOTEL_CLUSTER])
    a = a.drop(HOTEL_CLUSTER, axis=1)
    a = a.join(one_hot)

    a.to_pickle(PICKLE_DIR+'label'+ds_discrim+'.pkl')

def make_candidate_generation_labels(df, ds_discrim):
    a = make_labels(df, ds_discrim)
    a_lst = np.array_split(a, 3)
    label_lst = [ai.pivot_table(index=USER_ID, aggfunc=np.sum) for ai in a_lst]
    label = pd.concat(label_lst).sort_values(by=USER_ID)

    label.to_pickle(PICKLE_DIR+'candidate_generation_label'+ds_discrim+'.pkl')    

def make_users_embedding(df, ds_discrim):
    users_df = _select_specific_cols(df, list(USER_FEATURES))
    users_df = users_df.pivot_table(index=USER_ID)
    users_df.to_pickle(PICKLE_DIR+'users_df'+ds_discrim+'.pkl')

    user_id_to_embedding_idx_mapping = {k: v for v, k in enumerate(users_df.index)}
    pickle_it(user_id_to_embedding_idx_mapping, 'user_id_to_embedding_idx_mapping'+ds_discrim)

    overall_avg_user = users_df.mean()
    overall_avg_user.to_pickle(PICKLE_DIR+'overall_avg_user_series'+ds_discrim+'.pkl')

def make_search_embeddings(df, ds_discrim):
    # requires hotel cluster, so can only run on train and val
    search_df = _select_specific_cols(df, list(SEARCH_FEATURES) + ['hotel_cluster']) if ds_discrim == '_val' else pd.read_pickle(PICKLE_DIR+'search_df_train.pkl')
    search_df.fillna(0, inplace=True) # from preliminary analysis know that cols with na only exist for search features; replacing nas with 0 bc too many to just drop

    if ds_discrim == '_val':
        search_df.to_pickle(PICKLE_DIR+"search_df"+ds_discrim+'.pkl')
        print('finished writing pickle')

    print('calc mean')
    overall_avg_search = search_df.mean(numeric_only=True)
    print('write mean')
    overall_avg_search.to_pickle(PICKLE_DIR+"overall_avg_search_series"+ds_discrim+'.pkl')

    return search_df

def make_hotel_embeddings(df, ds_discrim):
    hotel_df = _select_specific_cols(df, list(HOTEL_FEATURES) + [USER_ID]).drop(columns=['hotel_cluster'])#.groupby('hotel_cluster').mean() #  fine that the ints get averaged into a float; since everything would be avg anyway
    hotel_df.to_pickle(PICKLE_DIR+'hotel_df'+ds_discrim+'.pkl')

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


if __name__ == '__main__':
    train_df = pd.read_pickle(PICKLE_DIR+'init_train_df_split.pkl')
    val_df = pd.read_pickle(PICKLE_DIR+'init_val_df_split.pkl')
    print('sets loaded')