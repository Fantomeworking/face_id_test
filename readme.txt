系统ubuntu18.10 
ide:sublime text3 
python3.7 
pytouch1.0.1 
cuda10.0 
cv2
内容未完善，请勿用于商业用途
1.cv2获取摄像头(0)信息(笔记本)
识别脸部信息（偷懒从cv2库中直接调用）
当识别到脸部出现在屏幕中央时获取图片信息3*200*200的脸部图片，一名用户共获取20张脸部图片信息
2.进入train.py中进行训练，储存并接与上一次图片信息尾部一起训练，第一份录入数据对应识别结果标识为1，以此类推
训练模式为一次cnn,3层卷积层200->100->50->25,层数3->9->18->36
CNN(
  (conv1): Sequential(
    (0): Conv2d(3, 9, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): Conv2d(9, 18, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv3): Sequential(
    (0): Conv2d(18, 36, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (out): Linear(in_features=22500, out_features=50, bias=True)
)
批训练，每批随机由Dataloader控制20个样本，训练100次
Adam加速器
激励函数
训练完成后储存所有脸部信息为 all_Face.npy 于当前目录下；储存用户信息为 user.json 于当前目录下；备份当前脸部数据为 当前用户名_Face.npy 于face_data文件中
保存训练结果，在test.py中进行测试。
缺点：受光线影响极大；人脸识别需要一个基本人脸库的支持；用户数量多起来时只能用分布式的方法解决；all_Face.npy储存占用极大，4张脸已经76M了；内存占用极大，cnn数据量过大时训练极慢；
优点：没有优点贼垃圾
警告：内容垃圾，请勿用于商业用途
下一步：hadoop通过pyspark储存和读取脸部数据
