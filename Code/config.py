
# set the path-to-files
TRAIN_FILE = "../../Data/advertisment/Raw/round1_ijcai_18_train_20180301.txt"
TEST_FILE = "../../Data/advertisment/Raw/round1_ijcai_18_test_a_20180301.txt"

SUB_DIR = "./output"


NUM_SPLITS = 5
RANDOM_SEED = 42

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    # 'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    # 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    # 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    # 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    # 'ps_car_10_cat', 'ps_car_11_cat',
]

NUMERIC_COLS = [
    'shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description'
]

IGNORE_COLS = [
    #keys
    'instance_id','user_id','context_id','is_trade','shop_id','item_id',
    #time
    'context_timestamp',
    #string category
    'item_category_list','item_property_list','predict_category_property'
    
]

KEYS = [
    'instance_id','user_id','context_id','is_trade','shop_id','item_id'
]

LABEL = 'is_trade'
