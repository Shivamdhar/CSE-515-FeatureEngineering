"""
Class for storing textual descriptors data
"""

class DescTxtStructure:

    def __init__(self, term_tf_df_tfidf, data_type):
        self.term = term_tf_df_tfidf[0]
        self.tf = float(term_tf_df_tfidf[1])
        self.df = float(term_tf_df_tfidf[2])
        self.tfidf = float(term_tf_df_tfidf[3])
        self.data_type = data_type
        
    def __str__(self):
        return("%s %s %s %s %s" % (self.term, self.tf, self.df, self.tfidf, self.data_type))
    
    def get_val(self, model):
        if model == "TF":
            return self.tf
        if model == "DF":
            return self.df
        if model == "TF-IDF":
            return self.tfidf
        else:
            return
    
    def get_data_type(self):
        return self.data_type
