バイブコーディングにて作成

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

1. AWS認証情報の設定。ローカルで実行する場合は作業ディレクトリ直下に.awsを作成し、credentials, configを配置
2. 必要なPythonパッケージのインストール
3. AWSリソースのデプロイ