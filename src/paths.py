from helpers import notexist_exit, create_if_not_exists

#-------------------------------------
class PATH(object):
    def __init__(self, fclayer, debug_mode=False, use_ngrams=False):
        debug='.debug' if debug_mode else ''
        ngrams='.ngrams' if use_ngrams else ''

        captions_path = '../captions'
        self.tr_captions_file = captions_path+'/train2014.sentences.txt'+debug+ngrams+'.bz2'
        self.val_caption_file = captions_path+'/Validation_val2014.sentences.txt'+debug+ngrams+'.bz2'
        self.test_caption_file= captions_path+'/Test_val2014.sentences.txt'+debug+ngrams+'.bz2'
        notexist_exit([self.tr_captions_file, self.val_caption_file, self.test_caption_file])

        fcx = "fc"+str(fclayer)
        visual_path = '../visualembeddings/' + fcx
        num_examples= '1' if debug_mode else '20'
        self.tr_visual_embeddings_file = visual_path+'/COCO_train2014_hybridCNN_'+fcx+'.sparse'+debug+'.txt'
        self.val_visual_embeddings_file = visual_path+'/COCO_val2014_hybridCNN_'+fcx+'.sparse.'+num_examples+'K_Validation'+debug+'.txt'
        self.test_visual_embeddings_file= visual_path+'/COCO_val2014_hybridCNN_'+fcx+'.sparse.'+num_examples+'K_Test'+debug+'.txt'
        notexist_exit([self.tr_visual_embeddings_file, self.val_visual_embeddings_file, self.test_visual_embeddings_file])

        pca_path = '../pca/'+fcx
        self.mean_file = pca_path+'/COCO_train2014_hybridCNN_'+fcx+'_ReLu_L2Norm_PC_from65536.mean.dat.txt'
        self.eigen_file = pca_path+'/COCO_train2014_hybridCNN_'+fcx+'_ReLu_L2Norm_PC_from65536.eigen.dat.txt'
        notexist_exit([self.mean_file, self.eigen_file])

        self.checkpoint_dir = create_if_not_exists('../models')
        self.predictions_dir = create_if_not_exists('../predictions')
        self.summaries_dir = create_if_not_exists('../summaries')
