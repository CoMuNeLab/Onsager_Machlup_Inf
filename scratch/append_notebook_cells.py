import json

notebook_path = 'Grid_Search_Analysis.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Weighted Network Error (MAE)\n",
        "Let's evaluate the mean absolute distance between the inferred and true non-zero edge weights."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "df['Weighted_MAE'] = pd.to_numeric(df['Weighted_MAE'], errors='coerce')\n",
        "\n",
        "g = sns.catplot(\n",
        "    data=df, \n",
        "    x=\"M_pairs\", y=\"Weighted_MAE\", hue=\"Regime\", col=\"dt_meas\", \n",
        "    col_wrap=3, kind=\"point\", height=4, aspect=1.2, capsize=.1,\n",
        "    palette={\"T\": \"orange\", \"S\": \"blue\"}, markers=[\"o\", \"s\"], linestyles=[\"-\", \"--\"]\n",
        ")\n",
        "g.set_axis_labels(\"Number of Samples (M)\", \"Mean Absolute Error (MAE)\")\n",
        "g.set_titles(r\"$\\Delta t$ = {col_name}\")\n",
        "plt.subplots_adjust(top=0.9)\n",
        "g.fig.suptitle('Weighted MAE vs M_pairs across dt_meas (Lower is Better)')\n",
        "plt.savefig('images/detailed/Weighted_MAE_vs_M_faceted.png', dpi=300, bbox_inches='tight')\n",
        "plt.show()"
    ]
}

nb['cells'].extend([markdown_cell, code_cell])

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=4)
    
print("Successfully added new cells to the notebook via JSON.")
