Thank you for looking at the code of my project. I hope you will enjoy the visualizations I made for my thesis. <br/>
<br/>
You are welcome to try all files. <br/>
<br/>
Below I have included guides to each file on how to use them:<br/>
<br/>

matrix.py ----------------------------------------------------------------------------------------------------------------------<br/>

This file does not run anything on its own. <br/>
<br/>
It is purely a library for matrix multiplication used in nn.py!<br/>
<br/>
So if you copy nn.py into your own IDE make sure to include this library otherwise nn.py won't work.<br/>
<br/>

nn.py ---------------------------------------------------------------------------------------------------------------------------<br/>

This file can be run to see how a neural network works when it is build from scratch. <br/>
<br/>
It was not the main project of the thesis, but it was my first stepping stone to understanding how to use neural networks in the main project.<br/>
<br/>
To use it simply run the file and it will start training on the XOR training data. <br/>
<br/>
Currently it is set to predict (1, 0), but any of the predictions can be un-commented to see the results.<br/>
<br/>

Thesis2025.py -----------------------------------------------------------------------------------------------------------------<br/>

This is the main file used for training rocket boosters to land and also to produce videos, pictures and graphs.<br/>
<br/>
You can try playing with the initial position of the rocket to see how it trains. <br/>
<br/>
In the file are two default starting positions for you to choose from, you can use these or make one yourself!<br/>
<br/>
You can play around with the code quite a lot. I have put in the code where it is fun to change stuff and see what happens. I don't recommend changing many things outside these options.<br/>
<br/>
It will start learning after you press run, the only thing you will see is the fitness being printed.<br/>
<br/>
After a successful landing (10,000 score) the code will say successful landing and a pygame window will be opened.<br/>
<br/>
It will automatically play generation 1. <br/>
<br/>
The rocket booster with the highest fitness of that generation will have a slightly blue colour. This shows the highest progress made by the system so far.<br/>
<br/>

Want to:<br/>
Replay?                                    press 'SPACE_BAR'<br/>
Go to next recorded generation?            press 'RIGHT_ARROW'<br/>
<br/>

If the successful landing was in generation 56, the 50th generation will be played, this is then followed by the landing of the best agent in generation 56.<br/>
<br/>
After closing this last pygame window, the first graph will pop up showing the agents states during this landing.<br/>
<br/>
After closing that graph, the final graph will pop up and this will show the learning process over all the generations.<br/>
<br/>

!!! Disclaimer !!!<br/>
<br/>
It is still a randomily initiated neural network, some evolution runs will take forever!<br/>
<br/>
My advice is any run above 200 generations, is to run it again with a new time seed.<br/>
<br/>
The rocket will always eventually land, but it could take forever because it is based on random chance.<br/>
<br/>







