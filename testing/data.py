class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.set_expected_outputs(x,y)

    def give_inputs(self):
        inputs=[self.x,self.y]
        return inputs
    
    def set_expected_outputs(self, x, y):#dynamic function that can change depending on what we want
        """
        if y > (x/2)**2 - 2:
            self.t=1
            self.f=0
        else:
            self.t=0
            self.f=1
        """

        """
        if y>x*-1:
            self.t=1
            self.f=0
        else:
            self.t=0
            self.f=1
        """
        
        
        if y<-x**4-x**3+3*x**2+2*x+4 and y>x**4/5-x**3+x/2:
            self.t=1
            self.f=0
        else:
            self.t=0
            self.f=1