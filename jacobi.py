import numpy as np
#condition suffisante pour la convergence
def lineJacobi(mat,i):
    sum = 0
    for j in range(len(mat)):
        if j!=i :
            sum += abs(mat[i][j])
        else : pass
    return sum<=abs(mat[i][i])
        
def isJacobi(mat):
    i = 0
    while (i<len(mat)):
        if not(lineJacobi(mat,i)):
            return False
        i+=1
    return True
        

m1 = [[2, -1],[-1, 2]]
m2 = [[2, -1],[-2, -1]]
m3 = [[2, -1, 0],[-1, 2,-1],[0,0, -1]]

print(lineJacobi(m2,0))
print(isJacobi(m3))