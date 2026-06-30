# TAMPURA 環境追加ガイド

`find_block_stack_dice` を実装する過程で得られた知見を2つのドキュメントにまとめた。

| ドキュメント | 対象 | 内容 |
|---|---|---|
| [pddl_design.md](pddl_design.md) | PDDL 設計の基礎 | プランナーの動作原理・前提条件の設計方法（特定のフレームワークに依存しない） |
| [tampura_internals.md](tampura_internals.md) | TAMPURA 実装の詳細 | フレームワーク内部の仕組み・実装上の落とし穴 |

新しい環境を追加するときは、まず `tampura_internals.md` でフレームワークの全体像を把握し、プランナーの挙動に問題があれば `pddl_design.md` を参照する。
