.PHONY: env kernel smoke panel

env:
	conda env create -f env.yml || true

kernel:
	python -m ipykernel install --user --name h191 --display-name "H191 (py311)"

smoke:
	python scripts/download.py --name demo_dataset --url https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv --dest-subdir raw --filename airtravel.csv --dataset-id DEMO_001 --notes "smoke test"

panel:
	python scripts/build_panel.py
