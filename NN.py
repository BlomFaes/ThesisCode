#Neural network class
#Bram Faes

import Matrix as m
import numpy as np
import math
import random
# import pygame

width = 900
heigth = 500
FPS = 60

# WIN = pygame.display.set_mode((width, heigth))
# pygame.display.set_caption("NN")

color1 = (255,255,255)
color2 = (0,0,0)

# def draw_nodes(node):
#     pygame.draw.circle(WIN, color1, (node.x, node.y), 30)
#
# def draw_lines(node1, node2, width):
#     #print(width)
#     pygame.draw.line(WIN,color2, (node1.x,node1.y),(node2.x,node2.y), round((width + 10)/1))



class NeuralNet(object):
    def __init__(self, inputs, hidden, outputs):
        self.info = [inputs, hidden, outputs]

        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs

        self.w_ih = m.matrix(self.hidden, self.inputs)
        self.w_ho = m.matrix(self.outputs, self.hidden)
        self.w_ih.randomize()
        self.w_ho.randomize()

        self.bias_h = m.matrix(self.hidden, 1)
        self.bias_o = m.matrix(self.outputs, 1)

        self.learningRate = 0.1

    def predict(self, input_array):
        inp = m.arrToMatrix(input_array)

        hid = m.mtrxMultiplyStatic(self.w_ih, inp)
        hid.addMtrx(self.bias_h)
        hid.map1(sigmoid)

        out = m.mtrxMultiplyStatic(self.w_ho, hid)
        out.addMtrx(self.bias_o)
        out.map1(sigmoid)

        return m.mtrxToArr(out)

    def train(self, input, goals):
        inputsObj = m.arrToMatrix(input)
        goalsObj = m.arrToMatrix(goals)

        hidden_outputs = m.mtrxMultiplyStatic(self.w_ih, inputsObj)
        hidden_outputs.addMtrx(self.bias_h)
        hidden_outputs.map1(sigmoid)

        finals = m.mtrxMultiplyStatic(self.w_ho, hidden_outputs)
        finals.addMtrx(self.bias_o)
        finals.map1(sigmoid)

        output_errors = m.subMtrx(goalsObj, finals)

        weights_ho_t = m.transpose(self.w_ho)
        hidden_errors = m.mtrxMultiplyStatic(weights_ho_t, output_errors)

        gradients = m.map2(finals, dsigmoid)
        gradients.scalarMatrix(output_errors)
        gradients.scalar(self.learningRate)

        hidden_t = m.transpose(hidden_outputs)
        weights_ho_delta = m.mtrxMultiplyStatic(gradients, hidden_t)

        self.w_ho.addMtrx(weights_ho_delta)
        self.bias_o.addMtrx(gradients)

        hidden_gradients = m.map2(hidden_outputs, dsigmoid)
        hidden_gradients.scalarMatrix(hidden_errors)
        hidden_gradients.scalar(self.learningRate)

        inputs_t = m.transpose(inputsObj)
        weights_ih_delta = m.mtrxMultiplyStatic(hidden_gradients, inputs_t)

        self.w_ih.addMtrx(weights_ih_delta)
        self.bias_h.addMtrx(hidden_gradients)

        error_value = mse(goalsObj, finals)
        if i % 1000 == 0:
            print("MSE:", error_value)

    # for i in range(len(self.input_nodes)):
    #     draw_nodes(self.input_nodes[i])
    # for i in range(len(self.hidden_nodes)):
    #     draw_nodes(self.hidden_nodes[i])
    # for i in range(len(self.output_nodes)):
    #     draw_nodes(self.output_nodes[i])
    #
    # for i in range(len(self.input_nodes)):
    #     for j in range(len(self.hidden_nodes)):
    #         draw_lines(self.input_nodes[i], self.hidden_nodes[j], self.w_ih.data[j][i])
    #
    # for i in range(len(self.hidden_nodes)):
    #     for j in range(len(self.output_nodes)):
    #         draw_lines(self.hidden_nodes[i], self.output_nodes[j], self.w_ho.data[j][i])

        #print(self.w_ih.data)
        #print(self.w_ho.data)

#------------------------------------------------------------------------------------

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def sigmoid2(x):
    return (1/1+np.exp(-x))*2-2

def dsigmoid(y):
    return y * (1 - y)

def mse(m, n):
    total = 0
    count = m.rows * m.cols
    for i in range(m.rows):
        for j in range(m.cols):
            diff = m.data[i][j] - n.data[i][j]
            total += diff * diff
    return total / count


#------------------------------------------------------------------------------------

# class node(object):
#     def __init__(self, layer, number, total):
#         self.layer = layer
#         self.number = number
#         self.total = total
#         self.x = (width/3*self.layer-150)
#         self.y = heigth*(self.number/(self.total+1))

class trainingData(object):
    def __init__(self, inputs, goals):
        self.inputs = inputs
        self.goals = goals

#------------------------------------------------------------------------------------


inputs = []
t1 = trainingData([1, 0],[1])
inputs.append(t1)
t2 = trainingData([0, 1],[1])
inputs.append(t2)
t3 = trainingData([0,0],[0])
inputs.append(t3)
t4 = trainingData([1,1],[0])
inputs.append(t4)

nn = NeuralNet(2, 4, 1)
for i in range(10000):
    for j in inputs:
        nn.train(j.inputs, j.goals)
print("Finished training!")

# print(nn.predict([0,1]))
print(nn.predict([1,0]))
#print(nn.predict([0,0]))
# print(nn.predict([1,1]))


#------------------------------------------------------------------------------------




# def update():
#     WIN.fill(100)
#     # draw_nodes(1,2)
#     # draw_nodes(2,5)
#     # draw_nodes(3,1)
#
#     #j = random.choice(inputs)
#     #nn.train(j.inputs, j.targets)
#     #print(nn.finals.data)
#     pygame.display.update()


# def main():
#     clock = pygame.time.Clock()
#     run = True
#     while run:
#         clock.tick(FPS)
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False
#                 #print(nn.predict([0,0]))
#         update()
#
#
#     pygame.quit()
#
# if __name__ == "__main__":
#     main()
