"""
Utility file for extracting features from our text.

Features we could maybe extract below

 * Word Count: Higher word count may indicate Extroversion and Judging. In contrast, lower word count could indicate Intuition and Sensing.
 * Words Per Sentence: Higher WPS might indicate higher Thinking.
 * Number of Social Words: Words like "friend", "talk", "they", "us", etc. might indicate extroversion.
 * Number of Personal Pronouns: Words like "I", "my", "me", etc. might indicate introversion.
 * Polarity: Strong positive or negative sentiment might indicate Feeling. In contrast, minimal polarity could indicate Thinking.
 * Subjectivity: High subjectivity might correlate with Feeling while low subjectivity could indicate Thinking..
 * Type-Token Ratio (TTR): The number of unique words divided by total number of words. A higher TTR could correlate with Intution and Thinking.
"""
from textblob import TextBlob

def split_tweets(text_col):
    """
    Purpose: Separate tweets column into a list of strings, effectively getting rid of the '|||' delimiter.
    
    Arguments:
        text_col (list of str): Column from table that contains every word, including '|||' separators for sentences.
    """
    tweets = []

    for string in text_col:
        separated_tweets = string.split('|||')
        tweets.append(separated_tweets)
    
    return tweets
    
def get_word_count(text_col):
    """
    Purpose: Calculates the total word count across the 50 tweets

    Arguments:
        text_col (list of lists of str): Column from table that contains every sentence as different elements of a list/
    
    Outputs:
        word_count (list of ints): Value representing total word count, does not include '|||' separators.
    """
    word_count_col = []

    for list_of_tweets in text_col:
        word_count = 0

        # Loop through inner list (each tweet)
        for tweet_text in list_of_tweets:
            # Strip any extra whitespace
            cleaned_text = tweet_text.strip()

            if cleaned_text:
                # Add to word count
                words = cleaned_text.split()
                word_count += len(words)

        word_count_col.append(word_count)
    
    return word_count_col

def get_polarity(text_col):
    """
    Purpose: Calculates the total polarity across the 50 tweets

    Arguments:
        text_col (list of lists of str): Column from table that contains every sentence as different elements of a list/
    
    Outputs:
        polarity (list of float): Value representing polarity of all 50 tweets
    """
    polarity_col = []

    # Loop through each instance
    for list_of_tweets in text_col:
        total_polarity = 0.0
        
        # Iterates through each post in row
        for tweet in list_of_tweets:
            sentiment = TextBlob(tweet).sentiment

            total_polarity += sentiment.polarity
        
        polarity_col.append(total_polarity / 50) # total polarity / num posts
    
    return polarity_col
            
def get_ttr(text_col):
    """
    Purpose: Calculates the Type-Token Ration (TTR) across the 50 tweets for each row
                TTR = (Unique Words / Total Words)

    Arguments:
        text_col (list of lists of str): Column from table that contains every sentence as different elements of a list/
    
    Outputs:
        ttr (list of floats): New column representing ttr for each row
    """
    ttr_col = []

    # Loop through each instance
    for list_of_tweets in text_col:
        all_tokens = []
        
        # Iterates through each post in row
        for tweet in list_of_tweets:
            words = TextBlob(tweet).words

            all_tokens.extend(words)
        
        total_tokens = len(all_tokens)

        lower_case_tokens = [token.lower() for token in all_tokens]
        unique_tokens = len(set(lower_case_tokens))

        ttr_col.append(unique_tokens / total_tokens)
    
    return ttr_col

def get_subjectivity(text_col):
    """
    Purpose: Calculates the subjectivity across the 50 tweets for each row

    Arguments:
        text_col (list of lists of str): Column from table that contains every sentence as different elements of a list
    
    Outputs:
        subjectivity (list of floats): New column representing subjectivity for each row
    """
    subjectivity_col = []

    # Loop through each instance
    for list_of_tweets in text_col:
        total_subjectivity = 0.0
        
        # Iterates through each post in row
        for tweet in list_of_tweets:
            sentiment = TextBlob(tweet).sentiment

            total_subjectivity += sentiment.subjectivity

        subjectivity_col.append(total_subjectivity / 50) # Total subjectivity / num posts (50)
    
    return subjectivity_col

def get_tag_ratios(text_col, word_count_col):
    """
    Purpose: Calculates the verb, adjective, personal pronoun, and third-person pronoun ratios.

    Arguments:
        text_col (list of lists of str): Column from table that contains every sentence as different elements of a list
        word_count_col (list of lists of ints): Column from table representing total word count of the 50 tweets per row
    
    Outputs:
        verb_ratio_col (list of ints): New column containing ratio of verbs per row
        adjective_ratio_col (list of ints): New column containing ratio of adjectives per row
        first_person_ratio_col (list of ints): New column containing ratio of personal pronouns per row
        third_person_ratio_col (list of ints): New column containing ratio of third-person pronouns per row
        adjective count
    """
    # Define verb tags and adjective tags
    VERB_TAGS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    ADJECTIVE_TAGS = {'JJ', 'JJR', 'JJS'}

    # Define basic pronouns
    FIRST_PERSON_PRONOUNS = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}
    THIRD_PERSON_PRONOUNS = {'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 
                             'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'}
    
    verb_ratio_col = []
    adjective_ratio_col = []
    first_person_ratio_col = []
    third_person_ratio_col = []

    # Loop through each row of tweets and word count
    for list_of_tweets, total_words in zip(text_col, word_count_col):
        total_verbs = 0
        total_adjectives = 0
        total_fp = 0 # total first-person words
        total_tp = 0 # total third-person words

        # Loop through each individual tweet
        for tweet in list_of_tweets:
            blob = TextBlob(tweet)

            # Iterate through the (word, tag) tuples finding pronouns as we go
            for word, tag in blob.tags:
                # Check if word is verb or adjective
                if tag in VERB_TAGS:
                    total_verbs += 1
                elif tag in ADJECTIVE_TAGS:
                    total_adjectives += 1
                
                # Check if word is generic pronoun
                if tag.startswith("PRP"):
                    word_lower = word.lower()

                    # Check if word matches our pronoun types
                    if word_lower in FIRST_PERSON_PRONOUNS:
                        total_fp += 1
                    elif word_lower in THIRD_PERSON_PRONOUNS:
                        total_tp += 1

        # Calculate ratios and append
        verb_ratio_col.append(total_verbs / total_words)
        adjective_ratio_col.append(total_adjectives / total_words)
        first_person_ratio_col.append(total_fp / total_words)
        third_person_ratio_col.append(total_tp / total_words)

    return verb_ratio_col, adjective_ratio_col, first_person_ratio_col, third_person_ratio_col
        

