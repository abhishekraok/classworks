import collections
def listoflists(replacelist,position,string):
    temp=replacelist#t[q0]
    #print string
    tem= [[ temp[0:position] ],[ temp[position:len(temp)], string[position:len(string)]  ]]
    replacelist=tem
    #print replacelist
    return replacelist

def match_string(s,ts): # s='xy', ts='xxx'
    j=0
    mat_len=0
    while j<len(s) and j<len(ts):
        print 'compare string', s[j], ts[j],s,ts
        if s[j]==ts[j]:
            mat_len+=1
        else:
            break
        j+=1
    return mat_len


def match(s,t):
    #s is the string xy, t is the array [xxx,[yy, [y z]],zzz]
    mat=-1
    for branch in t:
        if isinstance(branch, basestring):#branch is a string:
            print 'compare array', s[0],branch[0],s,branch
        
            if s[0]==branch[0]:
                mat=0
                mat_len=match_string(s,branch) # gives the length of string matched
                if mat_len<len(branch):
                    print 'len', mat_len,branch,mat_len,string
                    t.remove(branch)
                    branch=listoflists(branch,mat_len,string)
                    t.append(branch)
                    #t.replace(branch, branch_new)
                    print branch,t
                else:
                    match(s,branch[1])
                break
        else:
            match(s,branch)
            break
        
    print 'new', t
    return (t,mat)
            
s='ataaatg$'
tree=[s]
hold=[s]
m=len(s)
# Final #
i=1
while i<m:
    string=s[i:m]
    [tree,mat2]=match(string,tree)
    if mat2==-1: #No string matched.
            print 'New branch needed at root'
            #hold.append(string)
            print 'before adding',tree
            tree.append(string)
            print 'after adding',tree
    i+=1
print 't',tree
