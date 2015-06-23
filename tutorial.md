UFCORINの使い方
====
事前準備とビルド
----
[libsvm]{https://www.csie.ntu.edu.tw/~cjlin/libsvm/}をビルドして実行ファイルのあるディレクトリにパスを通し、`svm-predict`等でlibsvmが動かせるようにしておいてください。

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

このように表示されれば、予報実験が成功しています。出力中の
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

のように出力されています。


これらの出力はファイルからも見ることができます。デフォルトの設定では、予報戦略ファイルが存在したのと同じディレクトリに、以下のようなファイルが生成されます。

```
UFCORIN$ ls resource/sample-strategy-local*
resource/sample-strategy-local-regres.txt  resource/sample-strategy-local-session.yml
resource/sample-strategy-local-result.yml  resource/sample-strategy-local.yml
```

ここで、`-result.yml`は予報結果、`-session.yml`は予報の経過と結果が記録されているファイルで、`-regres.txt`には個々の時点での予報値と観測値が記録されています。
