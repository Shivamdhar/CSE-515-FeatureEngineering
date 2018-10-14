### class for storing textual descriptors data

class DescTxtStructure:

    def __init__(self, tag_tf_df_tfidf):
        self.tag = tag_tf_df_tfidf[0]
        self.tf = float(tag_tf_df_tfidf[1])
        self.df = float(tag_tf_df_tfidf[2])
        self.tfidf = float(tag_tf_df_tfidf[3])
        
    def __str__(self):
        return("%s %s %s %s" % (self.tag, self.tf, self.df, self.tfidf))
    
    def getVal(self, model):
        if model == "TF":
            return self.tf
        if model == "DF":
            return self.df
        if model == "TF-IDF":
            return self.tfidf
        else:
            return
