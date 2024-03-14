# ===============================
# Editor: Jingyuan Xu
# Create Date: 2024/3/4
# Description: The function of evaluating the search ability of the evolutionary algorithm 
# is encapsulated, and the bbob dataset of COCO is applied
# ===============================
from dependency import *

class EvaluateFunctions(object):
    # function 1: Sphere function
    def evaluateByF1(self, x):
        sum = 0
        for num in x:
            sum += num * num
        return sum
    
    # Schwefel's Problem 2.22
    def evaluateByF2(self, x):
        return np.sum(np.abs(x))+np.prod(np.abs(x));

    # Schwefel's Problem 1.2
    def evaluateByF3(self, x):
        dim=np.size(x)
        o=0
        sum = 0
        for i in range(dim):
            sum += x[i]
            o+=sum * sum;
        return o
    
    # Schwefel's Problem 2.21
    def evaluateByF4(self, x):
        return np.max(np.abs(x))
    
    # Generalized Rosenbrock's Function
    def evaluateByF5(self, x):
        dim = np.size(x)
        sum = 0
        for i in range(dim-1):
            sum += 100*(x[i+1] - x[i]*x[i])*(x[i+1] - x[i]*x[i]) + (x[i]-1)*(x[i]-1)
        return np.sum(100*np.power((x[1:dim]-(np.power(x[0:dim-1],2))),2)+(np.power(x[0:dim-1]-1,2)))
    
    # Step Function
    def evaluateByF6(self, x):
        return np.sum(np.power(np.abs((x+0.5)),2))
    
    # Quartic Function i.e. Noise
    def evaluateByF7(self, x):
        return np.sum(np.dot(np.arange(1,np.size(x)+1).T,(np.power(x,4))))+random.random();
    
    # Generalized Schwefel's Problem 2.26
    def evaluateByF8(self, x):
        c2 = np.reshape(x,[np.size(x),1])
        sum = -np.dot(np.sin(np.sqrt(np.abs(x))), c2)
        return sum
    
    # Generalized Rastrigin's Function
    def evaluateByF9(self, x):
        return np.sum(np.power(x,2)-10*np.cos((2*math.pi)*x))+10*np.size(x)
    
    # Ackley's Function
    def evaluateByF10(self, x):
        return -20*np.exp(-0.2*np.sqrt(np.sum(np.power(x,2))/np.size(x)))-np.exp(np.sum(np.cos((2*math.pi)*x))/np.size(x))+20+np.exp(1);
    
    # Generalized Griewank's Function
    def evaluateByF11(self, x):
        return np.sum(np.power(x,2))/4000-np.prod(np.cos(x/np.sqrt(np.arange(1,np.size(x)+1))))+1;

    # Generalized Penalized Function(No.1)
    def evaluateByF12(self, x):
        dim=len(x)
        pi=np.pi
        sum_1 = (np.pi/dim)*(10*((np.sin(pi*(1+(x[0]+1)/4)))**2)
                         +np.sum((((x[:dim-2]+1)/4)**2)*
                                 (1+10*((np.sin(pi*(1+(x[1:dim-1]+1)/4))))**2))
                         +((x[dim-1])/4)**2)
        sum_2 = np.sum(self.UFunc(x,10,100,4))
        O=sum_1 + sum_2
        return O
    # sub function: U function
    def UFunc(self, x,a,k,m):
        dim =len(x)
        U = np.zeros(dim)
        for i in range(len(x)):
            if x[i]>a:
                U[i]=k*((x[i]-a)**m)
            elif x[i]<-a:
                U[i]=k*((-x[i]-a)**m)
            else:
                U[i]=0
        return U
    
    # Generalized Penalized Function(No.2)
    def evaluateByF13(self, x):
        dim=len(x)
        pi=np.pi
        O=0.1*((np.sin(3*pi*x[0]))**2+np.sum(((x[0:dim-2])-1)**2*(1+(np.sin(3*pi*x[1:dim-1]))**2)))
        +((x[dim-1]-1)**2)*(1+(np.sin(2*pi*x[dim-1]))**2)+np.sum(self.UFunc(x,5,100,4))
        return O
    
    # Shekel's Foxholes Function
    def evaluateByF14(self, x):
        aS=np.array([[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],
                [-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]])
        bS=np.zeros(25)
        for j in range(0,25):
            bS[j]=np.sum((x.T-aS[:,j])**6)
        O=(1/500+np.sum(1/(np.arange(1,25+1)+bS)))**(-1)
        return O
    
    # Kowalik's Function
    def evaluateByF15(self, x):
        aK=np.array([0.1957,0.1947,0.1735,0.16,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
        bK=np.array([0.25,0.5,1,2,4,6,8,10,12,14,16])
        bK=1/bK
        O=np.sum((aK-((x[0]*(bK**2+x[1]*bK))/(bK**2+x[2]*bK+x[3])))**2)
        return O
    
    # Six-Hump Camel-Back Function
    def evaluateByF16(self, x):
        return 4*(x[0]**2)-2.1*(x[0]**4)+(x[0]**6)/3+x[0]*x[1]-4*(x[1]**2)+4*(x[1]**4)
    
    # Branin Function
    def evaluateByF17(self, x):
        pi=np.pi
        O=((x[1])-(x[0]**2)*5.1/(4*(pi**2))+5/pi*x[0]-6)**2+10*(1-1/(8*pi))*np.cos(x[0])+10
        return O
    
    # Goldstein-Price Function
    def evaluateByF18(self, x):
        return (1+((x[0]+x[1]+1)**2)*(19-14*x[0]+3*(x[0]**2)-14*x[1]+6*x[0]*x[1]+3*(x[1]**2)))*(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*(x[0]**2)+48*x[1]-36*x[0]*x[1]+27*(x[1]**2)))
    
    # Hartman's Family (No.1)
    def evaluateByF19(self, x):
        aH=np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
        cH=np.array([1,1.2,3,3.2])
        pH=np.array([[0.3689,0.117,0.2673],[0.4699,0.4387,0.747],
                    [0.1091,0.8732,0.5547],[0.03815,0.5743,0.8828]])
        O=0
        for i in range(0,4):
            O=O-cH[i]*np.exp(-(np.sum(aH[i]*((x-pH[i])**2))))
        return O
    
    # Hartman's Family (No.2)
    def evaluateByF20(self, x):
        aH=np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
        cH=np.array([1,1.2,3,3.2])
        pH=np.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],[0.2329,0.413,0.8307,0.3736,0.1004,0.9991],
        [0.2348,0.1415,0.3522,0.2883,0.3047,0.6650],[0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
        O=0
        for i in range(0,4):
            O=O-cH[i]*np.exp(-(np.sum(aH[i]*((x-pH[i])**2))))
        return O
    
    # Shekel's Family (No.1)
    def evaluateByF21(self, x):
        aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
                      [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
        cSH=np.array([[0.1],[0.2],[0.2],[0.4],[0.4],
                      [0.6],[0.3],[0.7],[0.5],[0.5]])
        O=0
        for i in range(0,5):
            O=O-(np.sum((x-aSH[i])**2)+cSH[i])**(-1)
        return O

    # Shekel's Family (No.2)
    def evaluateByF22(self, x):
        aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
                     [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
        cSH=np.array([[0.1],[0.2],[0.2],[0.4],[0.4],
                      [0.6],[0.3],[0.7],[0.5],[0.5]])
        O=0
        for i in range(0,7):
            O=O-(np.sum((x-aSH[i])**2)+cSH[i])**(-1)
        return O

    # Shekel's Family (No.3)
    def evaluateByF23(self, x):
        aSH=np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],
                      [2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
        cSH=np.array([[0.1],[0.2],[0.2],[0.4],[0.4],
                      [0.6],[0.3],[0.7],[0.5],[0.5]])
        O=0
        for i in range(0,10):
            O=O-(np.sum((x-aSH[i])**2)+cSH[i])**(-1)
        return O

    def getEvaluateFunc(self, funIndex):
        if funIndex==1:
            return bbobtorch.create_f01(30)
        elif funIndex==2:
            return self.evaluateByF2
        elif funIndex==3:
            return self.evaluateByF3
        elif funIndex==4:
            return self.evaluateByF4
        elif funIndex==5:
            return self.evaluateByF5
        elif funIndex==6:
            return self.evaluateByF6
        elif funIndex==7:
            return self.evaluateByF7
        elif funIndex==8:
            return self.evaluateByF8
        elif funIndex==9:
            return self.evaluateByF9
        elif funIndex==10:
            return self.evaluateByF10
        elif funIndex==11:
            return self.evaluateByF11
        elif funIndex==12:
            return self.evaluateByF12
        elif funIndex==13:
            return self.evaluateByF13
        elif funIndex==14:
            return self.evaluateByF14
        elif funIndex==15:
            return self.evaluateByF15
        elif funIndex==16:
            return self.evaluateByF16
        elif funIndex==17:
            return self.evaluateByF17
        elif funIndex==18:
            return self.evaluateByF18
        elif funIndex==19:
            return self.evaluateByF19
        elif funIndex==20:
            return self.evaluateByF20
        elif funIndex==21:
            return self.evaluateByF21
        elif funIndex==22:
            return self.evaluateByF22
        elif funIndex==23:
            return self.evaluateByF23