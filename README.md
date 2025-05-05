# Linux開発環境

このプロジェクトは、Dockerを使用したLinux開発環境を提供します。

## 必要条件

- Docker
- Docker Compose

## 使用方法

1. 環境の起動:
```bash
docker-compose up -d
```

2. コンテナへの接続:
```bash
docker-compose exec linux-dev bash
```

3. 環境の停止:
```bash
docker-compose down
```

## 含まれるツール

- Ubuntu 22.04
- build-essential
- curl
- git
- vim

## 注意事項

- 作業ディレクトリは `/workspace` にマウントされます
- コンテナ内での変更は、ホストマシンと同期されます 