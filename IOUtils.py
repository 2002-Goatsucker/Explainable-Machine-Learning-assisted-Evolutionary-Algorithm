from dependency import *

def write(path, info):
    with open(path,"w") as f:
        f.write(info)

def writeMatrix(path, matrix):
    info = ""
    with open(path,"w") as f:
        for line in matrix:
            for num in line:
                info+=str(num) + " "
            info += "\n"
        f.write(info)

def Matrix2DTorchSwitcher(matrix):
    matrix_torch = []
    for line in matrix:
        matrix_torch.append(torch.asarray(line).to(torch.float32))
    return matrix_torch