from Bio import SeqIO
import re
import Bio.Phylo
from Bio import Phylo
import numpy as np
from treetime import TreeAnc
import json
from .utils import write_json
import sys
import datetime as dt

def parse_beast_tree(data, tipMap, verbose=False):
    """
    data is a tree string, tipMap is a dict parsed from the nexus file
    Author: Gytis Dudas
    """
    i=0 ## is an adjustable index along the tree string, it is incremented to advance through the string
    stored_i=None ## store the i at the end of the loop, to make sure we haven't gotten stuck somewhere in an infinite loop

    cur_node = Bio.Phylo.Newick.Clade() ## new branch
    cur_node.name = 'root' ## start with root
    cur_node.clades = [] ## list of children
    node_count=0 ## node counter

    while i != len(data): ## while there's characters left in the tree string - loop away
        if stored_i == i and verbose==True:
            print('%d >%s<'%(i,data[i]))

        assert (stored_i != i),'\nTree string unparseable\nStopped at >>%s<<\nstring region looks like this: %s'%(data[i],data[i:i+5000]) ## make sure that you've actually parsed something last time, if not - there's something unexpected in the tree string
        stored_i=i ## store i for later

        if data[i] == '(': ## look for new nodes
            if verbose==True:
                print('%d adding node'%(i))
            node = Bio.Phylo.Newick.Clade() ## new object
            node.name = 'NODE_%07d'%(node_count) ## node name
            node.branch = 0.0 ## new node's branch length 0.0 for now
            node.up = cur_node ## new node's parent is current node
            node.clades = [] ## new node will have children
            node.attrs = {} ## initiate attrs dictionary
            cur_node.clades.append(node) ## add new node to children of current node
            cur_node = node ## new node is now current node
            node_count += 1 ## increment node counter
            i+=1 ## advance in tree string by one character

        numericalTip=re.match('(\(|,)([0-9]+)(\[|\:)',data[i-1:i+100]) ## look for tips in BEAST format (integers).
        if numericalTip is not None:
            if verbose==True:
                print('%d adding leaf (BEAST) %s'%(i,numericalTip.group(2)))
            node = Bio.Phylo.Newick.Clade() ## new object
            node.name = tipMap[numericalTip.group(2)] ## assign decoded name
            node.up = cur_node ## leaf's parent is cur_node
            node.attrs = {} ## initiate attrs dictionary
            cur_node.clades.append(node) ## assign leaf to children of parent
            cur_node = node ## cur_node is leaf

            i+=len(numericalTip.group(2)) ## advance in tree string by however many characters the tip is encoded

        alphaTip=re.match('(\(|,)(\'|\")*([A-Za-z\_\-\|\.0-9\?\/]+)(\'|\"|)(\[)*',data[i-1:i+200])  ## look for tips with unencoded names - if the tips have some unusual format you'll have to modify this
        if alphaTip is not None:
            if verbose==True:
                print('%d adding leaf (non-BEAST) %s'%(i,alphaTip.group(3)))
            node = Bio.Phylo.Newick.Clade() ## new object
            node.name = alphaTip.group(3) ## assign name
            node.up = cur_node ## leaf's parent is cur_node
            node.attrs = {} ## initiate attrs dictionary
            cur_node.clades.append(node) ## assign leaf to children of parent
            cur_node = node ## cur_node is leaf

            i+=len(alphaTip.group(3))+alphaTip.group().count("'")+alphaTip.group().count('"') ## advance in tree string by however many characters the tip is encoded

        multitypeNode=re.match('\)([0-9]+)\[',data[i-1:i+100]) ## look for multitype tree singletons.
        if multitypeNode is not None:
            if verbose==True:
                print('%d adding multitype node %s'%(i,multitypeNode.group(1)))
            i+=len(multitypeNode.group(1))

        commentBlock=re.match('(\:)*\[(&[A-Za-z\_\-{}\,0-9\.\%=\"\'\+!#]+)\]',data[i:])## look for MCC comments
        if commentBlock is not None:
            if verbose==True:
                print('%d comment: %s'%(i,commentBlock.group(2)))
            comment=commentBlock.group(2)
            numerics=re.findall('[,&][A-Za-z\_\.0-9]+=[0-9\-Ee\.]+',comment) ## find all entries that have values as floats
            strings=re.findall('[,&][A-Za-z\_\.0-9]+=["|\']*[A-Za-z\_0-9\.\+]+["|\']*',comment) ## strings
            treelist=re.findall('[,&][A-Za-z\_\.0-9]+={[A-Za-z\_,{}0-9\.]+}',comment) ## complete history logged robust counting (MCMC trees)
            sets=re.findall('[,&][A-Za-z\_\.0-9\%]+={[A-Za-z\.\-0-9eE,\"\_]+}',comment) ## sets and ranges
            figtree=re.findall('\![A-Za-z]+=[A-Za-z0-9#]+',comment) ## figtree comments, in case MCC was manipulated in FigTree

            for vals in strings: ## string states go here
                tr,val=vals.split('=') ## split into key and value
                tr=tr[1:] ## key has preceding & or ,
                if re.search('.*[^0-9\.eE].*',val) is not None: ## string regex can sometimes match floats (thanks to beast2), only allow values with at least one non-numeric character
                    if '+' in val: ## state was equiprobable with something else
                        equiprobable=val.split('+') ## get set of equiprobable states
                        val=equiprobable[np.random.randint(len(equiprobable))] ## DO NOT ALLOW EQUIPROBABLE DOUBLE ANNOTATIONS (which are in format "A+B")
                    cur_node.attrs[tr]=val.strip('"') ## assign value to attrs, strip "

            for vals in numerics: ## assign all parsed annotations to traits of current branch
                tr,val=vals.split('=') ## split each value by =, left side is name, right side is value
                tr=tr[1:] ## ignore preceding & or ,
                if 'prob' not in tr:
                    cur_node.attrs[tr]=float(val) ## assign float to attrs

#             for val in treelist:  ### enables parsing of complete history logger output from posterior trees
#                 tr,val=val.split('=')
#                 tr=tr[1:]
#                 completeHistoryLogger=re.findall('{([0-9]+,[0-9\.\-e]+,[A-Z]+,[A-Z]+)}',val)
#                 setattr(cur_node,'muts',[])
#                 for val in completeHistoryLogger:
#                     codon,timing,start,end=val.split(',')
#                     cur_node.muts.append('%s%s%s'%(start,codon,end))

            states={} ## credible sets will be stored here
            for vals in sorted(sets,key=lambda s:'.set.prob' in s.split('=')[0]): ## sort comments so sets come before set probabilities
                tr,val=vals.split('=') ## split comment into key and value
                tr=tr[1:] ## key has & or , in front

                if 'set' in tr: ## dealing with set
                    trait=tr.split('.set')[0] ## get trait name
                    if '.prob' not in tr: ## dealing with credible set
                        states[trait]=[v.strip('"') for v in val[1:-1].split(',')] ## store credible set
                    elif '.prob' in tr: ## dealing with probability set
                        probs=map(float,val[1:-1].split(',')) ## turn probability set into a list of floats
                        cur_node.attrs['%s_confidence'%(trait)]={t:p for t,p in zip(states[trait],probs)} ## create dictionary of state:probability

                elif 'range' in tr: ## range, best to ignore
                    pass
                    #cur_node.attrs[tr.replace('range','maxima')]=list(map(float,val[1:-1].split(','))) ## list of floats
                elif 'HPD' in tr and '_R_' not in tr: ## highest posterior densities, excluding Markov rewards
                    cur_node.attrs[tr.replace('95%_HPD','confidence')]=list(map(float,val[1:-1].split(','))) ## list of floats


            if len(figtree)>0:
                print('FigTree comment found, ignoring')

            i+=len(commentBlock.group()) ## advance in tree string by however many characters it took to encode comments

        nodeLabel=re.match('([A-Za-z\_\-0-9\.]+)(\:|\;)',data[i:])## look for old school node labels
        if nodeLabel is not None:
            if verbose==True:
                print('old school comment found: %s'%(nodeLabel.group(1)))
            cur_node.name=nodeLabel.group(1)
            i+=len(nodeLabel.group(1))

        branchLength=re.match('(\:)*([0-9\.\-Ee]+)',data[i:i+100]) ## look for branch lengths without comments
        if branchLength is not None:
            if verbose==True:
                print('adding branch length (%d) %.6f'%(i,float(branchLength.group(2))))
            setattr(cur_node,'branch_length',float(branchLength.group(2)))
            i+=len(branchLength.group()) ## advance in tree string by however many characters it took to encode branch length

        if data[i] == ',' or data[i] == ')': ## look for bifurcations or clade ends
            i+=1 ## advance in tree string
            cur_node = cur_node.up

        if data[i] == ';': ## look for string end
            return cur_node
            break ## end loop

def parse_nexus(
    tree_path,
    # tip_regex='\|([0-9]+\-[0-9]+\-[0-9]+)',
    # date_fmt='%Y-%m-%d',
    treestring_regex='tree [A-Za-z\_]+([0-9]+)',
    verbose=False
):
    """
    Author: Gytis Dudas
    """

    tipFlag=False
    tips={}
    tipNum=0
    tree=None

    if isinstance(tree_path,str): ## determine if path or handle was provided to function
        handle=open(tree_path,'r')
    else:
        handle=tree_path

    for line in handle: ## iterate over lines
        l=line.strip('\n')

        nTaxa=re.search('dimensions ntax=([0-9]+);',l.lower()) ## get number of tips that should be in tree
        if nTaxa is not None:
            tipNum=int(nTaxa.group(1))
            if verbose==True:
                print('File should contain %d taxa'%(tipNum))

        treeString=re.search(treestring_regex,l) ## search for line with the tree
        if treeString is not None:
            treeString_start=l.index('(') ## find index of where tree string starts
            tree=parse_beast_tree(l[treeString_start:],tipMap=tips,verbose=verbose) ## parse tree string
            if verbose==True:
                print('Identified tree string')

        if tipFlag==True: ## going through tip encoding block
            tipEncoding=re.search('([0-9]+) ([A-Za-z\-\_\/\.\'0-9 \|?]+)',l) ## search for key:value pairs
            if tipEncoding is not None:
                tips[tipEncoding.group(1)]=tipEncoding.group(2).strip('"').strip("'") ## add to tips dict
                if verbose==True:
                    print('Identified tip translation %s: %s'%(tipEncoding.group(1),tips[tipEncoding.group(1)]))
            elif ';' not in l:
                print('tip not captured by regex:',l.replace('\t',''))

        if 'translate' in l.lower(): ## tip encoding starts on next line
            tipFlag=True
        if ';' in l:
            tipFlag=False

    assert tree,'Tree not captured by regex'
    assert tree.count_terminals()==tipNum,'Not all tips have been parsed.'
    print("Success parsing BEAST nexus")
    return Phylo.BaseTree.Tree.from_clade(tree)

def fake_alignment(T):
    """
    fake alignment to appease treetime when only using it for naming nodes...
    This is lifted from refine.py and should be imported?
    """
    from Bio import SeqRecord, Seq, Align
    seqs = []
    for n in T.get_terminals():
        seqs.append(SeqRecord.SeqRecord(seq=Seq.Seq('ACGT'), id=n.name, name=n.name, description=''))
    aln = Align.MultipleSeqAlignment(seqs)
    return aln

def get_root_date_offset(tree):
    """
    years from most recent tip of the root
    """
    greatest_dist2root = 0
    for leaf in tree.get_terminals():
        if leaf.dist2root > greatest_dist2root:
            greatest_dist2root = leaf.dist2root;
    return greatest_dist2root

def decimalDate(date,date_fmt="%Y-%m-%d",variable=False,dateDelimiter='-'):
    """ Converts calendar dates in specified format to decimal date. """
    if variable==True: ## if date is variable - extract what is available
        dateL=len(date.split(dateDelimiter))
        if dateL==2:
            date_fmt=dateDelimiter.join(date_fmt.split(dateDelimiter)[:-1])
        elif dateL==1:
            date_fmt=dateDelimiter.join(date_fmt.split(dateDelimiter)[:-2])

    adatetime=dt.datetime.strptime(date,date_fmt) ## convert to datetime object
    year = adatetime.year ## get year
    boy = dt.datetime(year, 1, 1) ## get beginning of the year
    eoy = dt.datetime(year + 1, 1, 1) ## get beginning of next year
    return year + ((adatetime - boy).total_seconds() / ((eoy - boy).total_seconds())) ## return fractional year

def find_most_recent_tip(tree,regex="[0-9]{4}(\-[0-9]{2})*(\-[0-9]{2})*$",date_fmt="%Y-%m-%d",dateDelimiter='-'):
    """
    Search tip names using a regex (default: hyphen delimited numbers at the end of tip name) to identify dates, parse them as decimal dates and return the highest oneself.
    Can specify custom date formats in datetime notation (default: %Y-%m-%d), and different date delimiters (default: '-').
    """
    leaf_names=[leaf.name for leaf in tree.get_terminals()] ## get names of tips
    date_regex=re.compile(regex) ## regex pattern
    regex_matches=[date_regex.search(leaf) for leaf in leaf_names] ## search tips with regex
    assert regex_matches.count(None)==0,'These tip dates were not captured by regex %s: %s'%(regex,', '.join([leaf for leaf in leaf_names if date_regex.search(leaf)==None])) ## number of tips should match number of regex matches
    decimal_dates=[decimalDate(date_regex.search(leaf).group(),date_fmt=date_fmt,variable=True,dateDelimiter=dateDelimiter) for leaf in leaf_names] ## convert tip calendar dates to decimal dates

    return max(decimal_dates) ## return highest tip date

def collect_node_data(tree, root_date_offset, most_recent_tip_date):
    """
    A "normal" treetime example adds these traits:
        "branch_length": 0.0032664876882838745,
        "numdate": 2015.3901042843218,
        "clock_length": 0.0032664876882838745,
        "mutation_length": 0.003451507603103053,
        "date": "2015-05-23",
        "num_date_confidence": [
            2015.0320257687615,
            2015.6520676488697
        ]
    """

    data = {}
    root_date = most_recent_tip_date - root_date_offset
    for n in tree.find_clades():

        data[n.name] = {attr: n.attrs[attr] for attr in n.attrs if 'length' not in attr and 'height' not in attr} ## add all beast tree traits other than lengths and heights
        numeric_date = root_date + n.dist2root ## convert from tree height to absolute time
        data[n.name]['num_date'] = numeric_date ## num_date is decimal date of node
        data[n.name]['clock_length'] = n.branch_length ## assign beast branch length as regular branch length
        if n.is_terminal()==False:
            if 'height_confidence' in n.attrs:
                data[n.name]['num_date_confidence'] = [most_recent_tip_date - height for height in n.attrs['height_confidence']] ## convert beast 95% HPDs into decimal date confidences
        else:
            data[n.name]['posterior'] = 1.0 ## assign posterior of 1.0 to every tip (for aesthetics)

    return data

def computeEntropies(tree):
    """
    Computes entropies for discrete traits.
    """
    alphabets={} ## store alphabets
    for clade in tree.find_clades(): ## iterate over branches
        for attr in [key for key in clade.attrs if isinstance(clade.attrs[key],dict)]: ## iterate over branch attributes
            if attr in alphabets: ## if attr seen before
                for val in clade.attrs[attr]: ## iterate over attribute values of the node
                    if val not in alphabets[attr]: ## not seen this attribute value before
                        alphabets[attr].append(val)
            else:
                alphabets[attr]=[] ## not seen trait before - start a list of its values
                for val in clade.attrs[attr]: ## iterate over trait values for this branch
                    alphabets[attr].append(val)

    for clade in tree.find_clades(): ## iterate over branches
        for trait in alphabets: ## iterate over traits
            if trait in clade.attrs: ## branch has trait (in case there's a leaf-node difference in trait presence)
                trait_name=trait.split('_')[0] ## extract trait name root
                pdis=np.array([clade.attrs[trait][state] if state in clade.attrs[trait] else 0.0 for state in alphabets[trait]]) ## create state profile
                clade.attrs['%s_entropy'%(trait_name)] = -np.sum(pdis*np.log(pdis+1e-10)) ## compute entropy for trait

def print_suggested_config(nodes):
    def include_key(k):
        exclude_list = ["clock_length"]
        return (not k.endswith("_confidence") and not k.endswith("_entropy") and k not in exclude_list)
    attrs = set()
    for node in nodes:
        attrs.update({k for k in nodes[node].keys() if include_key(k)})

    print("\nA number of traits (node annotations) have been parsed. \
    The config file provided to `augur export` needs these added. \
    Here is a template block:")
    print("---------------------------------------------------------")
    def make_block(attr):
        if attr == "num_date":
            menuItem = "Sampling Date"
        else:
            menuItem = attr
        return {"menuItem": menuItem, "legendTitle": menuItem, "type": "continuous"}
    print(json.dumps({"color_options": {attr: make_block(attr) for attr in attrs}}, indent=2))
    print("---------------------------------------------------------")

def run(args):
    '''
    BEAST MCC tree to newick and node-data JSON for further augur processing / export
    '''
    print("importing from BEAST MCC tree", args.mcc)

    if args.recursion_limit:
        print("Setting recursion limit to %d"%(args.recursion_limit))
        sys.setrecursionlimit(args.recursion_limit)

    # node data is the dict that will be exported as json
    node_data = {
        'comment': "Imported from a BEAST MCC tree",
        'mcc_file': args.mcc
    }

    # parse the BEAST MCC tree
    tree = parse_nexus(tree_path=args.mcc, verbose=args.verbose)
    # Phylo.draw_ascii(tree)

    # the following commands are lifted from refine.py and mock it's behaviour when not calling treetime
    # importing from there may help prevent code divergence
    aln = fake_alignment(tree)
    # instantiate treetime for the sole reason to name internal nodes
    # note that tt.tree = T and this is modified in-place by this function
    tt = TreeAnc(tree=tree, aln=aln, ref=None, gtr='JC69', verbose=1)


    # time units need to be adjusted by the most recent tip date
    root_date_offset = get_root_date_offset(tree)
    print("root_date_offset:", root_date_offset, args.time_units)

    if args.most_recent_tip_date_fmt=='regex':
        if args.tip_date:
            most_recent_tip = find_most_recent_tip(tree,regex=args.tip_date)
        else:
            most_recent_tip = find_most_recent_tip(tree)
    elif args.most_recent_tip_date_fmt=='decimal':
        most_recent_tip = float(args.tip_date)

    # compute/extract the relevant data from nodes to be written to JSON for further augur processing
    computeEntropies(tree) ## compute entropies for discrete trait
    node_data['nodes'] = collect_node_data(tree, root_date_offset, most_recent_tip)

    # export very similarly to refine.py
    tree_success = Phylo.write(tree, args.output_tree, 'newick', format_branch_length='%1.8f')
    json_success = write_json(node_data, args.output_node_data)
    print("node attributes written to", args.output_node_data, file=sys.stdout)
    # import pdb; pdb.set_trace()

    print_suggested_config(node_data['nodes'])

    return 0 if (tree_success and json_success) else 1
