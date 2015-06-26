UFCORINの使い方
====
事前準備とビルド
----
[libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)をビルドして実行ファイルのあるディレクトリにパスを通し、`svm-predict`等でlibsvmが動かせるようにしておいてください。

[Glasgow Haskell Compiler(GHC)](https://www.haskell.org/ghc/) をインストールしてください。当方はGHC 7.8.4で動作検証しております。

このUFCORINレポジトリをチェックアウトし`make`を実行してください。うまくいけば、`./dist/build`以下に実行ファイルが作成されます。


ファイルの準備
----

UFCORINはAmazon AWS上もしくはローカルに存在するファイルを読み込んで動作します。ここでは、次のように、`forecast-features`以下にGOES X線フラックスの過去・未来データファイル、
`wavelet-features`以下にウェーブレット特徴量の時系列データファイルが展開されているものとします。

```sh
UFCORIN$ ls forecast-features/
backcast-goes-24.txt  backcast-hmi-24.txt  cast.zip              forecast-goes-48.txt  forecast-hmi-24.txt
backcast-goes-48.txt  backcast-hmi-48.txt  forecast-goes-0.txt   forecast-goes-72.txt  forecast-hmi-48.txt
backcast-goes-72.txt  backcast-hmi-72.txt  forecast-goes-24.txt  forecast-hmi-0.txt    forecast-hmi-72.txt
UFCORIN$ ls wavelet-features/
bsplC-301-N-0000-0000.txt  bsplC-301-S-0008-0001.txt  haarC-2-N-0000-0000.txt  haarC-2-S-0008-0001.txt
bsplC-301-N-0000-0001.txt  bsplC-301-S-0008-0002.txt  haarC-2-N-0000-0001.txt  haarC-2-S-0008-0002.txt
bsplC-301-N-0000-0002.txt  bsplC-301-S-0008-0004.txt  haarC-2-N-0000-0002.txt  haarC-2-S-0008-0004.txt
bsplC-301-N-0000-0004.txt  bsplC-301-S-0008-0008.txt  haarC-2-N-0000-0004.txt  haarC-2-S-0008-0008.txt
  ...
```

UFCORINの実行
----

予報機の実行ファイル名は `./dist/build/prediction-main/prediction-main` です。このプログラムは予報戦略ファイル名をコマンドライン引数にとって動作します。

レポジトリにはローカルファイルでの動作検証用の `./resource/sample-strategy-local.yml` という戦略ファイルが付属しています。このファイルを引数にとって予報機を動作させてみます。

```bash
UFCORIN$ ./dist/build/prediction-main/prediction-main ./resource/sample-strategy-local.yml 
using workdir: /tmp/spaceweather-10179
loading: file://./forecast-features/backcast-goes-24.txt
loading: file://./forecast-features/backcast-goes-48.txt
loading: file://./forecast-features/backcast-goes-72.txt
loading: file://./wavelet-features/haarC-2-S-0016-0016.txt
loading: file://./forecast-features/forecast-goes-24.txt
CVWeekly
testing: LibSVMOption {_libSVMType = 3, _libSVMKernelType = 2, _libSVMCost = 1.0, _libSVMEpsilon = Nothing, _libSVMGamma = 1.0e-2, _libSVMNu = Nothing, _libSVMAutomationLevel = 0, _libSVMAutomationPopSize = 10, _libSVMAutomationTolFun = 1.0e-3, _libSVMAutomationScaling = 2.0, _libSVMAutomationNoise = False}
.
...*
optimization finished, #iter = 3945
nu = 0.817046
obj = -2414.137160, rho = 7.371479
nSV = 6859, nBSV = 6848
Mean squared error = 0.270895 (regression)
Squared correlation coefficient = 0.408041 (regression)
sum TSS : 1.6677656609124325
tag: PredictionSuccess
predictionResultMap:
  MClassFlare:
    HeidkeSkillScore:
      scoreValue: 0.46654333690853855
      contingencyTable:
        TN: 6169
        TP: 854
        FN: 905
        FP: 430
      maximizingThreshold: -5.175680601517911
    TrueSkillStatistic:
      scoreValue: 0.4753657526107157
      contingencyTable:
        TN: 5084
        TP: 1240
        FN: 519
        FP: 1515
      maximizingThreshold: -5.430299911536409
  XClassFlare:
    HeidkeSkillScore:
      scoreValue: 0.2598171672432496
      contingencyTable:
        TN: 7642
        TP: 122
        FN: 132
        FP: 462
      maximizingThreshold: -4.975662158103695
    TrueSkillStatistic:
      scoreValue: 0.6494799884960085
      contingencyTable:
        TN: 7082
        TP: 197
        FN: 57
        FP: 1022
      maximizingThreshold: -5.159999999999998
  CClassFlare:
    HeidkeSkillScore:
      scoreValue: 0.46535781819181465
      contingencyTable:
        TN: 1266
        TP: 5398
        FN: 1060
        FP: 634
      maximizingThreshold: -5.834841586190942
    TrueSkillStatistic:
      scoreValue: 0.5429199198057082
      contingencyTable:
        TN: 1542
        TP: 4723
        FN: 1735
        FP: 358
      maximizingThreshold: -5.697608926221584

FN: ./resource/sample-strategy-local-result.yml
UFCORIN$
```

このように表示されれば、予報実験が成功しています。
出力中の
`sum TSS : 1.6677656609124325` という行が、X,≧M,≧Cクラスの予報のTrue Skill Statistic(TSS)の合計を表しています。
また、各クラスごとのTSSや、contingency tableの値も

```
  XClassFlare:
    TrueSkillStatistic:
      scoreValue: 0.6494799884960085
      contingencyTable:
        TN: 7082
        TP: 197
        FN: 57
        FP: 1022
```

のように出力されています。上記の出力からは、
XクラスのTSSは0.6494799884960085、
≧MクラスのTSSは0.4753657526107157、
≧CクラスのTSSは0.5429199198057082
であったことが読み取れます。



これらの出力はファイルからも見ることができます。デフォルトの設定では、予報戦略ファイルが存在したのと同じディレクトリに、以下のようなファイルが生成されます。

```
UFCORIN$ ls resource/sample-strategy-local*
resource/sample-strategy-local-regres.txt  resource/sample-strategy-local-session.yml
resource/sample-strategy-local-result.yml  resource/sample-strategy-local.yml
```

ここで、`-result.yml`は予報結果、`-session.yml`は予報の経過と結果が記録されているファイルで、`-regres.txt`には個々の時点での予報値と観測値が記録されています。

予報精度の分散の推定
--------
さて、この実験で表示された、上記のTSSの値はそのまま十数桁の精度をもつわけではありません。
これは特定の交差検定データセットに対して、この戦略が出したTSSにすぎず、
実用的な予報精度を見積もるにあたっては、未来の、未知の入力に対するTSSの値の確率分布を推定する必要があります。

このために、UFCORINには、入力データのなかで訓練データと試験データををシャッフルして予報実験を繰り返すための機能が備わっています。

- シャッフル交差検定データの乱数種の生成

`generate-survey`プログラムに自然数を引数として与えて実行すると、「良い」交差検定データのための乱数種を、指定された数だけ生成して出力します。

```
$ ./dist/build/generate-survey/generate-survey 100
using workdir: /tmp/spaceweather-2223
loading: /user/nushio/forecast/forecast-goes-24.txt
[(6371.0,6574.0),(1630.0,1709.0),(168.0,171.0)]
1275912768945169054
[(6589.0,6356.0),(1601.0,1738.0),(172.0,167.0)]
5194423306639535894
[(6399.0,6546.0),(1629.0,1710.0),(171.0,168.0)]
1149296450477154706
[(6352.0,6593.0),(1706.0,1633.0),(170.0,169.0)]
320668682659133633
...
```

ここで、「良い」データの基準としては、「X,≧M,≧Cクラスイベントのいずれについても、訓練データと試験データに含まれるイベント数の差が10%以内である」を採用しています。`generate-survey`プログラムは、「良い」乱数種に加えて、それぞれの種に対するイベント数の情報を出力します。訓練データと試験データに含まれるイベント数を揃えるのは、各イベントの予報精度を安定して測定するために必要です。

交差検定データの生成にこれらの乱数種を使用させている例が、`resource/sample-strategy-local-S1.yml`  および`resource/sample-strategy-local-S2.yml`です。私の環境で実行させてみた結果は以下のとおりです。
２つのファイルに記述されている予報戦略は、利用した入力時系列データやその重みなども含めてまったく同一のものです。
同一の予報戦略であっても、交差検定データにシャッフルを施せば、予報のTSSがばらつくことがわかっていただけるかと思います。

````
$ ./dist/build/prediction-main/prediction-main resource/sample-strategy-local-S1.yml 
using workdir: /tmp/spaceweather-5639
loading: file://./forecast-features/backcast-goes-24.txt
loading: file://./forecast-features/backcast-goes-48.txt
loading: file://./forecast-features/backcast-goes-72.txt
loading: file://./wavelet-features/haarC-2-S-0016-0016.txt
loading: file://./forecast-features/forecast-goes-24.txt
CVShuffled 1275912768945169054 CVWeekly
testing: LibSVMOption {_libSVMType = 3, _libSVMKernelType = 2, _libSVMCost = 1.0, _libSVMEpsilon = Nothing, _libSVMGamma = 1.0e-2, _libSVMNu = Nothing, _libSVMAutomationLevel = 0, _libSVMAutomationPopSize = 10, _libSVMAutomationTolFun = 1.0e-3, _libSVMAutomationScaling = 2.0, _libSVMAutomationNoise = False}
.
....*
optimization finished, #iter = 4064
nu = 0.815953
obj = -2430.922411, rho = 7.891013
nSV = 6807, nBSV = 6795
Mean squared error = 0.250155 (regression)
Squared correlation coefficient = 0.390421 (regression)
sum TSS : 1.7326988701457469
...
````


````
nushio@nushio16k:UFCORIN$ ./dist/build/prediction-main/prediction-main resource/sample-strategy-local-S2.yml 
using workdir: /tmp/spaceweather-5851
loading: file://./forecast-features/backcast-goes-24.txt
loading: file://./forecast-features/backcast-goes-48.txt
loading: file://./forecast-features/backcast-goes-72.txt
loading: file://./wavelet-features/haarC-2-S-0016-0016.txt
loading: file://./forecast-features/forecast-goes-24.txt
CVShuffled 5194423306639535894 CVWeekly
testing: LibSVMOption {_libSVMType = 3, _libSVMKernelType = 2, _libSVMCost = 1.0, _libSVMEpsilon = Nothing, _libSVMGamma = 1.0e-2, _libSVMNu = Nothing, _libSVMAutomationLevel = 0, _libSVMAutomationPopSize = 10, _libSVMAutomationTolFun = 1.0e-3, _libSVMAutomationScaling = 2.0, _libSVMAutomationNoise = False}
.
....*
optimization finished, #iter = 4054
nu = 0.815912
obj = -2424.489143, rho = 7.554115
nSV = 6829, nBSV = 6822
Mean squared error = 0.254871 (regression)
Squared correlation coefficient = 0.392881 (regression)
sum TSS : 1.7628628775025483
...
````

このように、予報戦略のTSSの推定には、交差検定データの選び方からくる分散がかならず含まれています。予報戦略のTSSを評価するにあたっては、TSSを単一の値と捉えるのではなく、かならず十分な数の「良い」乱数種を生成してTSS測定を繰り返し、TSSの分布を把握するようにしてください。


