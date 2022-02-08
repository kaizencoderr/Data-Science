'''
**************** PLEASE READ ***************

Script that reads in spam and ham messages and converts each training example
into a feature vector

Code intended for UC Berkeley course CS 189/289A: Machine Learning

Requirements:
-scipy ('pip install scipy')

To add your own features, create a function that takes in the raw text and
word frequency dictionary and outputs a int or float. Then add your feature
in the function 'def generate_feature_vector'

The output of your file will be a .mat file. The data will be accessible using
the following keys:
    -'training_data'
    -'training_labels'
    -'test_data'

Please direct any bugs to kevintee@berkeley.edu
'''

from collections import defaultdict
import glob
import re
import scipy.io
import numpy as np
from spellchecker import SpellChecker

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

# ************* Features *************

# Features that look for certain words
def freq_pain_feature(text, freq):
    return float(freq['pain'])

def freq_private_feature(text, freq):
    return float(freq['private'])

def freq_bank_feature(text, freq):
    return float(freq['bank'])

def freq_money_feature(text, freq):
    return float(freq['money'])

def freq_drug_feature(text, freq):
    return float(freq['drug'])

def freq_spam_feature(text, freq):
    return float(freq['spam'])

def freq_prescription_feature(text, freq):
    return float(freq['prescription'])

def freq_creative_feature(text, freq):
    return float(freq['creative'])

def freq_height_feature(text, freq):
    return float(freq['height'])

def freq_featured_feature(text, freq):
    return float(freq['featured'])

def freq_differ_feature(text, freq):
    return float(freq['differ'])

def freq_width_feature(text, freq):
    return float(freq['width'])

def freq_other_feature(text, freq):
    return float(freq['other'])

def freq_energy_feature(text, freq):
    return float(freq['energy'])

def freq_business_feature(text, freq):
    return float(freq['business'])

def freq_message_feature(text, freq):
    return float(freq['message'])

def freq_volumes_feature(text, freq):
    return float(freq['volumes'])

def freq_revision_feature(text, freq):
    return float(freq['revision'])

def freq_path_feature(text, freq):
    return float(freq['path'])

def freq_meter_feature(text, freq):
    return float(freq['meter'])

def freq_memo_feature(text, freq):
    return float(freq['memo'])

def freq_planning_feature(text, freq):
    return float(freq['planning'])

def freq_pleased_feature(text, freq):
    return float(freq['pleased'])

def freq_record_feature(text, freq):
    return float(freq['record'])

def freq_out_feature(text, freq):
    return float(freq['out'])

# Features that look for certain characters
def freq_semicolon_feature(text, freq):
    return text.count(';')

def freq_dollar_feature(text, freq):
    return text.count('$')

def freq_sharp_feature(text, freq):
    return text.count('#')

def freq_exclamation_feature(text, freq):
    return text.count('!')

def freq_para_feature(text, freq):
    return text.count('(')

def freq_bracket_feature(text, freq):
    return text.count('[')

def freq_and_feature(text, freq):
    return text.count('&')

# --------- Add your own feature methods ----------
def example_feature(text, freq):
    return int('example' in text)
    
def freq_carrot_feature(text, freq):
    return text.count('>')
    
def freq_carrot2_feature(text, freq):
    return text.count('<')

def freq_colon_feature(text, freq):
    return text.count(':')
    
def length_feature(text, freq):
    return len(text)
    
def freq_slash_feature(text, freq):
    return text.count('/')
    
def freq_percent_feature(text, freq):
    return text.count('%')
    
def freq_question_feature(text, freq):
    return text.count('?')
    
def freq_equal_feature(text, freq):
    return text.count('=')
    
def freq_plus_feature(text, freq):
    return text.count('+')
    
def freq_period_feature(text, freq):
    return text.count('.')
    
def freq_free_feature(text, freq):
    return float(freq['free'])
    
def freq_save_feature(text, freq):
    return float(freq['save'])
    
def freq_earn_feature(text, freq):
    return float(freq['earn'])
    
def freq_cash_feature(text, freq):
    return float(freq['cash'])
    
def freq_winner_feature(text, freq):
    return float(freq['winner'])
    
def freq_nocost_feature(text, freq):
    return float(freq['no cost'])
    
def freq_numcaps_feature(text, freq):
    return sum(1 for c in text if c.isupper())
    
def freq_you_feature(text, freq):
    return float(freq['you'])
    
def freq_your_feature(text, freq):
    return float(freq['your'])
    
def freq_re_feature(text, freq):
    return float(freq['re'])
    
def freq_apply_feature(text, freq):
    return float(freq['apply'])
    
def freq_call_feature(text, freq):
    return float(freq['call'])
    
def freq_buy_feature(text, freq):
    return float(freq['buy'])
    
def freq_limited_feature(text, freq):
    return float(freq['limited'])
    
def freq_time_feature(text, freq):
    return float(freq['time'])
    
def freq_help_feature(text, freq):
    return float(freq['help'])
    
def freq_email_feature(text, freq):
    return float(freq['email'])
    
def freq_address_feature(text, freq):
    return float(freq['address'])
    
def freq_info_feature(text, freq):
    return float(freq['information'])
    
def freq_order_feature(text, freq):
    return float(freq['order'])
    
def freq_use_feature(text, freq):
    return float(freq['use'])
    
def freq_file_feature(text, freq):
    return float(freq['file'])
    
def freq_link_feature(text, freq):
    return float(freq['link'])
    
def freq_fast_feature(text, freq):
    return float(freq['fast'])
    
def freq_no_feature(text, freq):
    return float(freq['no'])
    
def freq_dear_feature(text, freq):
    return float(freq['dear'])
    
def freq_friend_feature(text, freq):
    return float(freq['friend'])
    
def freq_ad_feature(text, freq):
    return float(freq['ad'])
    
def freq_increase_feature(text, freq):
    return float(freq['increase'])
    
def freq_trial_feature(text, freq):
    return float(freq['trial'])
    
def freq_subscribe_feature(text, freq):
    return float(freq['subscribe'])
    
def freq_cure_feature(text, freq):
    return float(freq['cure'])
    
def freq_weight_feature(text, freq):
    return float(freq['weight'])

def freq_loss_feature(text, freq):
    return float(freq['loss'])
    
def freq_join_feature(text, freq):
    return float(freq['join'])
    
def freq_garunteed_feature(text, freq):
    return float(freq['garunteed'])
    
def freq_millions_feature(text, freq):
    return float(freq['millions'])
    
def freq_gift_feature(text, freq):
    return float(freq['gift'])
    
def freq_member_feature(text, freq):
    return float(freq['member'])
    
def freq_getaway_feature(text, freq):
    return float(freq['getaway'])
    
def freq_cancel_feature(text, freq):
    return float(freq['cancel'])
    
def freq_risk_feature(text, freq):
    return float(freq['risk'])
    
def freq_promise_feature(text, freq):
    return float(freq['promise'])
    
def freq_warranty_feature(text, freq):
    return float(freq['warranty'])
    

# Generates a feature vector
def generate_feature_vector(text, freq):
    feature = []
    feature.append(freq_pain_feature(text, freq))
    feature.append(freq_private_feature(text, freq))
    feature.append(freq_bank_feature(text, freq))
    feature.append(freq_money_feature(text, freq))
    feature.append(freq_drug_feature(text, freq))
    feature.append(freq_spam_feature(text, freq))
    feature.append(freq_prescription_feature(text, freq))
    feature.append(freq_creative_feature(text, freq))
    feature.append(freq_height_feature(text, freq))
    feature.append(freq_featured_feature(text, freq))
    feature.append(freq_differ_feature(text, freq))
    feature.append(freq_width_feature(text, freq))
    feature.append(freq_other_feature(text, freq))
    feature.append(freq_energy_feature(text, freq))
    feature.append(freq_business_feature(text, freq))
    feature.append(freq_message_feature(text, freq))
    feature.append(freq_volumes_feature(text, freq))
    feature.append(freq_revision_feature(text, freq))
    feature.append(freq_path_feature(text, freq))
    feature.append(freq_meter_feature(text, freq))
    feature.append(freq_memo_feature(text, freq))
    feature.append(freq_planning_feature(text, freq))
    feature.append(freq_pleased_feature(text, freq))
    feature.append(freq_record_feature(text, freq))
    feature.append(freq_out_feature(text, freq))
    feature.append(freq_semicolon_feature(text, freq))
    feature.append(freq_dollar_feature(text, freq))
    feature.append(freq_sharp_feature(text, freq))
    feature.append(freq_exclamation_feature(text, freq))
    feature.append(freq_para_feature(text, freq))
    feature.append(freq_bracket_feature(text, freq))
    feature.append(freq_and_feature(text, freq))
    feature.append(freq_carrot_feature(text, freq))
    feature.append(freq_carrot2_feature(text, freq))
    feature.append(freq_colon_feature(text, freq))
    feature.append(length_feature(text, freq))
    feature.append(freq_slash_feature(text, freq))
    feature.append(freq_percent_feature(text, freq))
    feature.append(freq_free_feature(text, freq))
    feature.append(freq_earn_feature(text, freq))
    feature.append(freq_cash_feature(text, freq))
    feature.append(freq_winner_feature(text, freq))
    feature.append(freq_nocost_feature(text, freq))
    feature.append(freq_numcaps_feature(text, freq))
    feature.append(freq_you_feature(text, freq))
    feature.append(freq_your_feature(text, freq))
    feature.append(freq_re_feature(text, freq))
    feature.append(freq_apply_feature(text, freq))
    feature.append(freq_call_feature(text, freq))
    feature.append(freq_buy_feature(text, freq))
    feature.append(freq_limited_feature(text, freq))
    feature.append(freq_question_feature(text, freq))
    feature.append(freq_equal_feature(text, freq))
    feature.append(freq_plus_feature(text, freq))
    feature.append(freq_period_feature(text, freq))
    feature.append(freq_time_feature(text, freq))
    feature.append(freq_help_feature(text, freq))
    feature.append(freq_email_feature(text, freq))
    feature.append(freq_address_feature(text, freq))
    feature.append(freq_info_feature(text, freq))
    feature.append(freq_order_feature(text, freq))
    feature.append(freq_use_feature(text, freq))
    feature.append(freq_file_feature(text, freq))
    feature.append(freq_link_feature(text, freq))
    feature.append(freq_fast_feature(text, freq))
    feature.append(freq_no_feature(text, freq))
    feature.append(freq_dear_feature(text, freq))
    feature.append(freq_friend_feature(text, freq))
    feature.append(freq_ad_feature(text, freq))
    feature.append(freq_increase_feature(text, freq))
    feature.append(freq_trial_feature(text, freq))
    feature.append(freq_subscribe_feature(text, freq))
    feature.append(freq_cure_feature(text, freq))
    feature.append(freq_weight_feature(text, freq))
    feature.append(freq_loss_feature(text, freq))
    feature.append(freq_join_feature(text, freq))
    feature.append(freq_garunteed_feature(text, freq))
    feature.append(freq_millions_feature(text, freq))
    feature.append(freq_member_feature(text, freq))
    feature.append(freq_gift_feature(text, freq))
    feature.append(freq_warranty_feature(text, freq))
    feature.append(freq_getaway_feature(text, freq))
    feature.append(freq_cancel_feature(text, freq))
    feature.append(freq_risk_feature(text, freq))
    feature.append(freq_promise_feature(text, freq))


    # --------- Add your own features here ---------
    # Make sure type is int or float

    return feature

# This method generates a design matrix with a list of filenames
# Each file is a single training example
def generate_design_matrix(filenames):
    design_matrix = []
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                text = f.read() # Read in text from file
            except Exception as e:
                # skip files we have trouble reading.
                continue
            text = text.replace('\r\n', ' ') # Remove newline character
            words = re.findall(r'\w+', text)
            word_freq = defaultdict(int) # Frequency of all words
            for word in words:
                word_freq[word] += 1

            # Create a feature vector
            feature_vector = generate_feature_vector(text, word_freq)
            design_matrix.append(feature_vector)
    return design_matrix

# ************** Script starts here **************
# DO NOT MODIFY ANYTHING BELOW

spam_filenames = glob.glob(BASE_DIR + SPAM_DIR + '*.txt')
spam_design_matrix = generate_design_matrix(spam_filenames)
ham_filenames = glob.glob(BASE_DIR + HAM_DIR + '*.txt')
ham_design_matrix = generate_design_matrix(ham_filenames)
# Important: the test_filenames must be in numerical order as that is the
# order we will be evaluating your classifier
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
test_design_matrix = generate_design_matrix(test_filenames)

X = spam_design_matrix + ham_design_matrix
Y = np.array([1]*len(spam_design_matrix) + [0]*len(ham_design_matrix)).reshape((-1, 1))

file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_design_matrix
scipy.io.savemat('spam_data.mat', file_dict)
