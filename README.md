## Guo‘s motions and emotions

  

​	运行环境：python3

​	由于源码没有封装得很好，因此使用步骤比较多，下面一一说明：

 

### 数据处理

1. 将数据集中的8个压缩包（除LIRIS原数据以外，其中训练集4个测试集4个）解压到根目录中的data文件夹下。

2. 装载Dev数据集：main.py中dataArg的visual_features一项为需要提取的特征，设置好后在根目录下输入：

   ```shell
   python main.py -m=data
   ```

   待程序运行结束后，将data文件夹下的uploaded_data文件夹改名为dev_uploaded_data。

3. 装载Test数据集：首先将dataArg的三个参数进行修改：

   ```python
   	'feature_dir' : "MEDIAEVAL17-TestSet-Visual_features/visual_features",
   	'valence_arousal_annotation_dir' : "MEDIAEVAL17-TestSet-Valence_Arousal-annotations/annotations",
   	'fear_annotation_dir' : "MEDIAEVAL17-TestSet-Fear-annotations/annotations",
   ```

   然后在utils/dataset.py中，将109行改为：

   ```python
   annotationPath = os.path.join(self.valenceArousalAnnotationDir, videoName+"_valence_arousal.txt")
   ```

   130行改为：

   ```python
   annotationPath = os.path.join(self.fearAnnotationDir, videoName+"_fear.txt") 
   ```

   之后的步骤和dev数据一样。最后将数据保存至test_uploaded_data中。



### LSTM

​	

1. Valance-Arousal：直接输入python main.py -m=train即可开始训练。训练过程中，每个epoch后会出现如下信息：![微信截图_20180106204523](E:\Motion in Movies\微信截图_20180106204523.png)

   其中的四个数字分别表示训练集的loss、r以及测试集的loss、r。

2. Fear：将dataArg和tf.app.flags中的label_name均改为"f"，然后运行main.py即可。不过为了加强效果，建议修改一些参数：keep_prob改为0.2，learning_rate=0.001，视频特征数据采用fc6。和上一条类似，Fear在每个epoch后共会产生8行结果，分别表示dev和test数据集上的accuracy、precision、recall、F1score。

   ​