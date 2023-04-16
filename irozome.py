import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Read the Excel file
df = pd.read_excel("古濱様_藍染データ(230408).xlsx", sheet_name="反射率等")

values_wave = list(range(360, 740, 10))  # 波長
cols = [f"{v}nm" for v in values_wave]
d = df[cols]
x = d.values

# データ名をtitleに変更
df = df.rename(columns={"データ名": "title"})

# titleからmaterialとconditionに分割
df["material"] = df["title"].str.split("_", expand=True)[1]
df["condition"] = df["title"].str.split("_", expand=True)[0]
labels = df[["material", "condition"]]

# "material"ごとに異なる色を割り当てるためのカラーマップ
material_colors = {material: px.colors.qualitative.Plotly[index % len(px.colors.qualitative.Plotly)] for index, material in enumerate(labels["material"].unique())}
material_colors

# グラフの初期化
fig = go.Figure()

# 既に凡例に追加されたmaterialをトラッキングするためのセット
added_materials = set()

# グラフの描画
for material in labels["material"].unique():
    _labels = labels[labels["material"] == material]
    for _x, _label in zip(x[_labels.index], _labels["condition"]):
        # showlegendオプションで重複する凡例を1つにまとめる
        showlegend = material not in added_materials
        fig.add_trace(go.Scatter(x=values_wave, y=_x, 
                                 name=material, 
                                 line=dict(color=material_colors[material], width=1), 
                                 legendgroup=material, showlegend=showlegend,
                                 hovertemplate=f"Condition: {_label}<br>Wavelength (nm): %{{x}}<br>Value: %{{y}}"))
        added_materials.add(material)

#　タイトル設定
fig.update_layout(title="古濱先生_スペクトルデータ_インド藍生葉染め, 化学建て")        
        
# 凡例設定
fig.update_layout(legend=dict(title="素材"))

# グラフサイズ設定
fig.update_layout(width=800, height=600)

#テンプレート指定
fig.update_layout(template="plotly_white")

# x軸設定
fig.update_xaxes(title_text="波長 (nm)", range=[350, 750])

# y軸設定
fig.update_yaxes(title_text="反射率 (%)")

# グラフ表示

# グラフの初期化
fig2 = go.Figure()

# マテリアルごとに3D散布図を描画
for material in df["material"].unique():
    material_data = df[df["material"] == material]
    fig2.add_trace(go.Scatter(
        x=material_data ["a*(D65)"],
        y=material_data ["b*(D65)"],
        mode="markers",
        marker=dict(size=6, color=material_colors[material], opacity=0.8),
        name=material,
        text=material_data["material"],
        customdata=material_data["condition"],
        
        hovertemplate="Material: %{text}<br>Condition: %{customdata}<br>a: %{x}<br>b: %{y}"
    ))

#　タイトル設定
fig2.update_layout(title="古濱先生_a*b*色度図データ_インド藍生葉染め, 化学建て")        
        
# 凡例設定
fig2.update_layout(legend=dict(title="素材"))

# x軸設定
fig2.update_xaxes(title_text="a*(D65)", range=[-20,30])

# y軸設定
fig2.update_yaxes(title_text="b*(D65)", range=[-40, 15])

# グラフサイズ設定
fig2.update_layout(width=800, height=600)

#テンプレート指定
fig2.update_layout(template="plotly_white")

# グラフ表示


import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# PCAを使用して3次元データを2次元に圧縮
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df[['L*(D65)', 'a*(D65)', 'b*(D65)']])
df['x_pca'] = X_pca[:, 0]
df['y_pca'] = X_pca[:, 1]

# t-SNEを使用して3次元データを2次元に圧縮
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(df[['L*(D65)', 'a*(D65)', 'b*(D65)']])
df['x_tsne'] = X_tsne[:, 0]
df['y_tsne'] = X_tsne[:, 1]

# UMAPを使用して3次元データを2次元に圧縮
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(df[['L*(D65)', 'a*(D65)', 'b*(D65)']])
df['x_umap'] = X_umap[:, 0]
df['y_umap'] = X_umap[:, 1]

# 可視化用の関数
def visualize_2d(df, x_col, y_col, title):
    fig3 = go.Figure()

    for material in df["material"].unique():
        material_data = df[df["material"] == material]
        fig3.add_trace(go.Scatter(
            x=material_data[x_col],
            y=material_data[y_col],
            mode="markers",
            marker=dict(size=6, opacity=0.8),
            name=material,
            text=material_data["material"],
            customdata=material_data["condition"],
            hovertemplate="Material: %{text}<br>Condition: %{customdata}<br>x: %{x}<br>y: %{y}"
        ))

    fig3.update_layout(template="plotly_white",
        title=title,
        xaxis_title=f"{x_col}",
        yaxis_title=f"{y_col}",
    )

    

# PCAの結果を可視化
visualize_2d(df, 'x_pca', 'y_pca', 'PCA')

# t-SNEの結果を可視化
visualize_2d(df, 'x_tsne', 'y_tsne', 't-SNE')

# UMAPの結果を可視化
visualize_2d(df, 'x_umap', 'y_umap', 'UMAP')

import pandas as pd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

#ファイル読込み
df1 = pd.read_excel("藍染めデータベース.xlsx", sheet_name="rawdata")

df1=df1[df1["location"]=="徳島矢野工場"]

# "material"ごとに異なる色を割り当てるためのカラーマップ
material_colors1 = {'ポリエステル': '#636EFA', 
                    'シルク': '#EF553B', 
                    'アクリル': '#00CC96', 
                    'レーヨン': '#AB63FA', 
                    'ウール': '#FFA15A', 
                    'アセテート': '#19D3F3', 
                    'ナイロン': '#FF6692', 
                    'コットン': '#B6E880', 
                    'ラミー': '#7FDBFF', 
                    'ポリエステル/ポリウレタン': '#FECB52', 
                    'ポリエステル/コットン': '#636EFA', 
                    'トリアセテート': '#f68c1f', 
                    'ジアセテート': '#e062a7', 
                    'リセヨル': '#59a1d3', 
                    'キュプラ': '#c85200'}

# グラフの初期化
fig4 = go.Figure()

# マテリアルごとに3D散布図を描画
for material in df1["material"].unique():
    material_data = df1[df1["material"] == material]
    fig4.add_trace(go.Scatter(
        x=material_data ["D65-10deg-a"],
        y=material_data ["D65-10deg-b"],
        mode="markers",
        marker=dict(size=6, color=material_colors1[material], opacity=0.8),
        name=material,
        text=material_data["material"],
        customdata=material_data["title"],
        
        hovertemplate="Material: %{text}<br>Condition: %{customdata}<br>a: %{x}<br>b: %{y}"
    ))

#　タイトル設定
fig4.update_layout(title="八木_a*b*色度図データ_徳島県矢野工場天然灰汁発酵建て")        
        
# 凡例設定
fig4.update_layout(legend=dict(title="素材"))

# x軸設定
fig4.update_xaxes(title_text="a*(D65)", range=[-20, 30])

# y軸設定
fig4.update_yaxes(title_text="b*(D65)", range=[-40, 15])

# グラフサイズ設定
fig4.update_layout(width=800, height=600)

#テンプレート指定
fig4.update_layout(template="plotly_white")


# グラフ表示


import base64
import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output


# 画像の準備
image_filename = r'C:\Users\kibos\OneDrive\デスクトップ\自炊ブックス\藍データ解析\色染め社パッチテスト_徳島天然灰汁発酵建て6回染め.jpeg'  # ここに画像ファイル名を入力
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app = dash.Dash(__name__)

server= app.server

app.layout = html.Div([
    html.Div([
        dcc.Graph(id="fig", figure=fig)
    ], style={'width': '50%', 'display': 'inline-block'}),
    html.Div([
        dcc.RadioItems(
            id='graph_selection',
            options=[
                {'label': 'PCA', 'value': 'PCA'},
                {'label': 't-SNE', 'value': 't-SNE'},
                {'label': 'UMAP', 'value': 'UMAP'}
            ],
            value='PCA',
            style={'display': 'inline-block'},
            labelStyle={'display': 'inline-block'}
        ),
        dcc.Graph(id="fig3")
    ], style={'width': '50%', 'display': 'inline-block','vertical-align': 'top'}),
    html.Div([
        dcc.Graph(id="fig2", figure=fig2)
    ], style={'width': '50%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id="fig4", figure=fig4)
    ], style={'width': '50%', 'display': 'inline-block'}),
    html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
        ], style={'width': '50%', 'display': 'inline-block'})
    ])


@app.callback(
    Output("fig3", "figure"),
    [Input("graph_selection", "value")]
)
def update_fig3(graph_selection):
    if graph_selection == 'PCA':
        return visualize_2d(df, 'x_pca', 'y_pca', 'PCA')
    elif graph_selection == 't-SNE':
        return visualize_2d(df, 'x_tsne', 'y_tsne', 't-SNE')
    elif graph_selection == 'UMAP':
        return visualize_2d(df, 'x_umap', 'y_umap', 'UMAP')


def visualize_2d(df, x_col, y_col, title):
    fig3 = go.Figure()

    for material in df["material"].unique():
        material_data = df[df["material"] == material]
        fig3.add_trace(go.Scatter(
            x=material_data[x_col],
            y=material_data[y_col],
            mode="markers",
            marker=dict(size=6, opacity=0.8),
            name=material,
            text=material_data["material"],
            customdata=material_data["condition"],
            hovertemplate="Material: %{text}<br>Condition: %{customdata}<br>x: %{x}<br>y: %{y}"
        ))

    fig3.update_layout(template="plotly_white",
                       title=title,
                       xaxis_title=f"{x_col}",
                       yaxis_title=f"{y_col}",
                       )

    return fig3


if __name__ == '__main__':
    app.run_server(debug=True)
