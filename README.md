# ML Pipeline on AWS

このプロジェクトは、AWS上で動作する機械学習パイプラインの実装です。

## アーキテクチャ

- **データ収集**: ECR + Fargate
- **前処理**: SageMaker Processing Job
- **学習**: SageMaker Training Job (GPU)
- **生成**: 
  - バッチ処理: SageMaker Batch Transform
  - API: SageMaker Endpoint

## プロジェクト構造

```
.
├── src/
│   ├── downloader/          # データ収集用コード
│   ├── preprocessor/        # 前処理用コード
│   ├── trainer/            # 学習用コード
│   └── generator/          # 推論用コード
├── infrastructure/         # AWSリソース定義
├── scripts/               # デプロイメントスクリプト
└── tests/                # テストコード
```

## セットアップ

1. AWS認証情報の設定
2. 必要なPythonパッケージのインストール
3. AWSリソースのデプロイ

## 使用方法

各コンポーネントの使用方法は、それぞれのディレクトリ内のREADMEを参照してください。

## 注意事項

- 各コンポーネントは独立して実行可能です
- 作業ディレクトリは `/workspace` にマウントされます
- コンテナ内での変更は、ホストマシンと同期されます
- GPUを使用する場合は、NVIDIA Container Toolkitがインストールされている必要があります

## トラブルシューティング

### 依存関係の問題

依存関係に問題がある場合は、各コンポーネントの`requirements.txt`を確認し、必要に応じて更新してください。

### メモリ不足

GPUのメモリが不足する場合は、以下の対策を試してください：
- バッチサイズを小さくする
- モデルのサイズを小さくする
- CPUモードで実行する

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。 