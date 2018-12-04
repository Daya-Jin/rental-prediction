队伍成员：

@zywu2018

@QQ435248055



因个人原因，train.csv与test.csv均作过处理，与网站上提供的原数据不同；队友代码中所用数据也跟我的不同，运行代码时需要注意。

更改后的特征映射关系为：

|       原特征       | 更改后的特征 |
| :----------------: | :----------: |
|        时间        |     Time     |
|       小区名       | Neighborhood |
| 小区房屋出租屋数量 |   RentRoom   |
|        楼层        |    Height    |
|       总层数       |  TolHeight   |
|      房屋面积      |   RoomArea   |
|      房屋朝向      |   RoomDir    |
|      居住状态      |  RentStatus  |
|      卧室数量      |   Bedroom    |
|      厅的数量      |  Livingroom  |
|      卫的数量      |   Bathroom   |
|      出租方式      |   RentType   |
|         区         |    Region    |
|        位置        |    BusLoc    |
|      地铁线路      |  SubwayLine  |
|      地铁站点      |  SubwaySta   |
|        距离        |  SubwayDis   |
|      装修情况      |  RemodCond   |
|       月租金       |    Rental    |



XGB: 1.94		

LGB: 1.86		

LGB+STA: 1.82		

LGB+XGB+STA: 1.80