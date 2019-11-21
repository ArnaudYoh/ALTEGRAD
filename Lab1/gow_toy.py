import string
from nltk.corpus import stopwords

#import os
#os.chdir() # to change working directory to where functions live
# import custom functions
from Lab1.library import clean_text_simple, terms_to_graph, core_dec

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

my_doc = 'A method for solution of systems of linear algebraic equations \
with m-dimensional lambda matrices. A system of linear algebraic \
equations with m-dimensional lambda matrices is considered. \
The proposed method of searching for the solution of this system \
lies in reducing it to a numerical system of a special kind.'

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc, my_stopwords=stpwds, punct=punct)

g = terms_to_graph(my_tokens, 4)

# number of edges
print("Number of edges:",len(g.es))

# the number of nodes should be equal to the number of unique terms
assert(len(g.vs) == len(set(my_tokens)))

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

print("Edge weights:", edge_weights)

for w in range(2, 21):
    gg = terms_to_graph(my_tokens, w)
    print("Density for window of size {}: {}".format(w, gg.density()))

g = terms_to_graph(my_tokens, 4)

# decompose g
core_numbers = core_dec(g,False)
print("Core numbers:", core_numbers)

if False:

    ### fill the gap (compare 'core_numbers' with the output of the .coreness() igraph method) ###

    # retain main core as keywords
    max_c_n = max(core_numbers.values())
    keywords = [kwd for kwd, c_n in core_numbers.items() if c_n == max_c_n]
    print("Keywords:", keywords)
