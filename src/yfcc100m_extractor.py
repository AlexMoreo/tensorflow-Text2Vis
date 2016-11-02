import bz2, sys
import urllib
from os import listdir
from os.path import isfile, join
from nltk.stem import WordNetLemmatizer

def extract_user_tags(yfcc100m_dir, lemmatize=False):
    lemmatizer = WordNetLemmatizer() if lemmatize else None
    lemma_posfix = '_lemma' if lemmatize else ''
    print('Extracting user_tags from %s' %yfcc100m_dir)
    yfcc100m_parts = [f for f in listdir(yfcc100m_dir) if isfile(join(yfcc100m_dir, f))]
    yfcc100m_parts.sort()

    with bz2.BZ2File(yfcc100m_dir+'user_tags'+lemma_posfix+'.bz2', 'w') as fout:
        for part in yfcc100m_parts:
            print('\t%s' % join(yfcc100m_dir, part))
            for line in bz2.BZ2File(join(yfcc100m_dir, part), 'r'):
                user_tags = line.split('\t')[8].strip()
                if user_tags:
                    user_tags = urllib.unquote_plus(user_tags).split(',')
                    if lemmatize:
                        lem_tags = []
                        for tag in user_tags:
                            try:
                                lem_tags.append(lemmatizer.lemmatize(tag))
                            except UnicodeDecodeError:
                                None
                        user_tags=lem_tags
                    if user_tags:
                        fout.write(' '.join(user_tags) + '\n')

#extract_user_tags('../wordembeddings/YFCC100M', lemmatize=True)
#extract_user_tags('../wordembeddings/YFCC100M', lemmatize=False)
