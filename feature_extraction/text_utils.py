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
        word_count (int): Value representing total word count, does not include '|||' separators.
    """
    word_count_col = []

    for list_of_tweets in text_col:
        word_count = 0

        # Loop through inner list (each tweet)
        for tweet_text in list_of_tweets:
            # 1. Strip any extra whitespace
            tweet_text.strip()

            # Add to word count
            word_count += len(tweet_text)

        word_count_col.append(word_count)
    
    return word_count_col

            
        