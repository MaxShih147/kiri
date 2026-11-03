# kiri

把一個 3D 模型沿著它「自然的接縫」自動切成幾個**封閉**的零件：每一塊都沒有開口、沒有非流形邊，可以直接拿去 3D 列印。不需要任何 3D 軟體，全部用 Python 完成。

- 網頁介面：拖入模型、拉滑桿看分塊、按一下切割、下載 ZIP
- 命令列：`kiri scan`、`kiri cut`、`kiri web`

## 安裝與使用

需要 [uv](https://docs.astral.sh/uv/)。

```sh
uv sync
uv run kiri web                     # 開啟網頁介面（http://localhost:8765/）
uv run kiri scan model.stl          # 看建議分幾塊
uv run kiri cut model.stl -k 6      # 切成 6 塊，輸出到 model_pieces/
uv run kiri cut model.stl --zip     # 用建議塊數，輸出成一個 ZIP
```

macOS 也可以直接雙擊 `Kiri.command` 開啟網頁介面。

支援讀取 STL、OBJ、PLY、GLB/glTF、OFF、3MF（透過 trimesh）；輸出為 STL。

## 流程

```
模型 ──► 減面 ──► 厚度(SDF) + 凹度 ──► 區域合併 ──► 選塊數 ──► 邊界修正(最小割)
                                                                     │
      ZIP ◄── 碎屑歸併 ◄── 封面 ◄── 依號誌距離場切開 ◄── 接縫平滑 ◄──┘
```

1. **減面**（`features.coarse`）：把模型減到 2,500–15,000 面，用於分析。二次誤差度量減面。[Garland & Heckbert 1997]
2. **形狀直徑函數 SDF**（`features.thickness`）：每個面往內打 30 條射線，射線分布在 120° 的圓錐內。去掉偏離中位數超過一個標準差的射線，依角度加權平均，再做對數正規化（α = 4）。手臂和軀幹的厚度差距很大，交界就是好的切點。[Shapira et al. 2008]
3. **凹度**（`features.concavity`）：相鄰面的角度距離 η(1 − cos θ)。凸邊打折計算（η = 0.2），所以凹的摺痕比凸的稜線更像切口。[Katz & Tal 2003]
4. **區域合併**（`segment`）：一開始每個面自成一區，每次合併「交界最不像切口」的兩個相鄰區域，並記下完整的合併順序。所以任何塊數都只是重播這段歷史的前段，網頁上拖滑桿是即時的。合併代價由三項相乘：
   - 對比：交界的平均凹度，加上兩側厚度差
   - 大小：Ward 變異增量 aA·aB/(aA+aB)，小區域會先被吸收 [Ward 1963]
   - 包圍度：交界佔較小區域周長的比例，幾乎被包住的區域只是碎片

   **建議塊數**取合併代價跳升最多的位置。
5. **邊界修正**（`refine`）：在每兩塊的交界附近三圈面內建一張圖，用最小割把邊界移到最凹的摺痕上。邊的容量是 1/(1 + 角度距離/平均)，乘上邊長。[Katz & Tal 2003]
6. **接縫平滑**（`seam`）：邊界環重新取樣，沿環做高斯低通濾波，再投影回表面，重複數次。邊界從鋸齒狀變成平順的曲線。
7. **切開**（`cut`）：在頂點上建一個到接縫的帶號距離場，再用 marching triangles 沿零等值線切開每個三角形。每條被切到的邊只產生一個新點，兩側共用，所以切口完全吻合。之後會清掉場中的孤島，避免切出碎泡。
8. **封面**（`cap`）：切口環先投影到最適平面；若投影後自我相交，改依弧長攤到圓上（圓盤參數化，[Tutte 1963; Floater 1997]）。接著用 Triangle 做品質三角化（邊界不加點）[Shewchuk 1996]。內部點再以調和插值放回 3D（像肥皂膜繃在鐵絲上），兩側共用同一張面、方向相反。
9. **逐塊剝離**（`pipeline`）：每次切下面積最小的一塊。剩下的部分若斷成幾段，就各自繼續處理。某一刀若讓結果不封閉或體積變負，就減少平滑次數再試，最後退回沿原始邊界直接切。太小的碎殼併入最近的零件。

最近點查詢（`nearest`）是自寫的 BVH（numba 平行）；點到三角形的最近點採用 Voronoi 區域法 [Ericson 2005]。

## 參考文獻

- L. Shapira, A. Shamir, D. Cohen-Or. *Consistent mesh partitioning and skeletonisation using the shape diameter function.* The Visual Computer 24(4), 2008.
- S. Katz, A. Tal. *Hierarchical mesh decomposition using fuzzy clustering and cuts.* ACM Transactions on Graphics 22(3) (SIGGRAPH), 2003.
- J. H. Ward. *Hierarchical grouping to optimize an objective function.* Journal of the American Statistical Association 58(301), 1963.
- M. Garland, P. S. Heckbert. *Surface simplification using quadric error metrics.* SIGGRAPH 1997.
- J. R. Shewchuk. *Triangle: Engineering a 2D quality mesh generator and Delaunay triangulator.* Applied Computational Geometry, LNCS 1148, 1996.
- W. T. Tutte. *How to draw a graph.* Proceedings of the London Mathematical Society 13, 1963.
- M. S. Floater. *Parametrization and smooth approximation of surface triangulations.* Computer Aided Geometric Design 14(3), 1997.
- C. Ericson. *Real-Time Collision Detection.* Morgan Kaufmann, 2005（§5.1.5 點到三角形的最近點）。

## 開發

```sh
uv run pytest                        # 測試：BVH 精確度、分塊、每塊封閉、體積守恆
uv run python scripts/make_sample.py # 重新產生 samples/figure.stl
```

## 授權

MIT，見 `LICENSE`。相依套件各有其授權；其中 Triangle 函式庫的授權禁止未經許可的商業使用，若要商用請先確認。
