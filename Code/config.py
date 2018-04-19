
# set the path-to-files
TRAIN_FILE = "../../Data/advertisment/Raw/round1_ijcai_18_train_20180301.txt"
TEST_FILE = "../../Data/advertisment/Raw/round1_ijcai_18_test_b_20180418.txt"

SUB_DIR = "./output"


NUM_SPLITS = 5
RANDOM_SEED = 42

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    'item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
    'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
    'user_age_level', 'user_star_level','context_page_id', 'hour', 'shop_id', 'shop_review_num_level',
    'shop_star_level','item_category_list_bin0','item_category_list_bin1','item_category_list_bin2'
    
]

STAT_DICT = {
    'item_id':['item_price_level', 'item_sales_level','item_collected_level', 'item_pv_level'],
    'user_id':['user_age_level', 'user_star_level'],
    'contest_id':['context_page_id'],
    'shop_id':['shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service',
               'shop_score_delivery','shop_score_description']
}

NUMERIC_COLS = [
    'shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description'
]

IGNORE_COLS = [
    #target
    'is_trade','cnt_rec',
    #keys
    'instance_id','user_id','context_id',
    #time
    'context_timestamp','day','time',
    #string category
    'item_category_list','predict_category_property','item_property_list','item_property_list_clean'
    
]

KEYS = [
    'instance_id','user_id','context_id','is_trade','shop_id','item_id'
]

LABEL = 'is_trade'
