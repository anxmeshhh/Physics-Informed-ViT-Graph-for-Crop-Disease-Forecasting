"""The literature survey, and how each research gap is answered by this system.

Every row of the survey carries three things:

``gap``
    What the cited work does not do - taken from the project's own related-work
    table.
``answer``
    The specific component of this implementation that closes that gap.
``demo``
    A *runnable* demonstration id. The frontend calls ``/api/demo/<id>`` and gets
    live output from the trained system, so a reviewer can see the claim being
    honoured rather than just asserted.

This is what turns the survey from a table into an argument.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Paper:
    no: int
    title: str
    authors: str
    method: str
    contribution: str
    gap: str
    answer: str
    demo: str
    module: str


PAPERS: tuple[Paper, ...] = (
    Paper(
        1, "Plant Disease Detection using Vision Transformers on Multispectral "
           "Natural Environment Images", "—",
        "Vision Transformer on multispectral plant images",
        "Transformer-based plant disease recognition under natural field conditions",
        "Visual detection only; no inter-field spatial relationships and no "
        "climate-informed forecasting.",
        "We attach every leaf to a real farm and date, build a spatio-temporal "
        "graph across farms, and forecast 1-7 days ahead.",
        "graph_structure", "graph/build.py",
    ),
    Paper(
        2, "Real-Time Plant Disease Identification: Fusion of Vision Transformer "
           "and Conditional Convolutional Network", "—",
        "ViT + conditional CNN + C3GAN augmentation",
        "Hybrid visual learning with generative augmentation for real-time ID",
        "Image-based identification only; no propagation across farms and no "
        "physics-informed climate constraints.",
        "Directed downwind edges model spore transport, and a physics loss "
        "enforces agronomic constraints during training.",
        "contagion", "physics/contagion.py",
    ),
    Paper(
        3, "Crop Disease and Pest Classification Using Swin Transformer and "
           "Dual-Attention Multi-Scale Fusion Network", "—",
        "Swin Transformer + dual-attention multi-scale fusion",
        "Multi-scale feature learning for disease and pest classification",
        "Classification-oriented; performs no spatial, climate-adaptive "
        "disease forecasting.",
        "A multi-horizon forecaster predicts risk at 1, 3, 5 and 7 days from "
        "fused visual, climatic and spatial evidence.",
        "multi_horizon", "models/heads.py",
    ),
    Paper(
        4, "Lightweight Vision Transformer with Lite-AVPSO Hyperparameter "
           "Optimization for Agricultural Disease Recognition", "—",
        "Lightweight ViT + hyperparameter optimisation",
        "Improves efficiency of transformer-based disease recognition",
        "Targets efficient recognition rather than climate variables, graph "
        "relations or early disease-spread forecasting.",
        "The backbone is frozen and its embeddings cached, so the whole system "
        "trains in minutes on a 6 GB laptop GPU while still using climate and graph.",
        "backbone_benchmark", "models/backbones.py",
    ),
    Paper(
        5, "Plant Disease Detection Using an Innovative Swin-Axial Transformer", "—",
        "Swin + Axial Transformer",
        "Advanced transformer architecture for plant disease detection",
        "Mainly image-based detection; lacks spatial field interaction and "
        "environmental disease modelling.",
        "Farm-to-farm interaction is explicit in the graph, and the "
        "environmental model is a published epidemiological knowledge base.",
        "epidemiology", "physics/epidemiology.py",
    ),
    Paper(
        6, "CycleGAN-Based Data Augmentation with CNN and Vision Transformers "
           "for Maize Leaf Disease Classification", "—",
        "CycleGAN augmentation + CNN + ViT",
        "Addresses data scarcity by generating additional training images",
        "Addresses scarcity and classification, but not climate conditions or "
        "transmission between fields.",
        "Rather than synthesising images we ground real ones in real "
        "meteorology, which is what gives the climate branch signal to learn.",
        "climate_niche", "data/assign.py",
    ),
    Paper(
        7, "A Comparison of Two Transformers in the Study of Plant Disease "
           "Classification", "—",
        "Comparative transformer analysis",
        "Evaluates transformer models for plant disease classification",
        "Limited to comparison and classification; no graph-based spatial "
        "reasoning or climate-aware forecasting.",
        "We benchmark four transformer families under one identical downstream "
        "pipeline, and then go beyond classification to forecasting.",
        "backbone_benchmark", "scripts/03_benchmark_backbones.py",
    ),
    Paper(
        8, "Abnormal Image Classification Using Vision Transformer for Smart "
           "Agriculture", "—",
        "Vision Transformer",
        "Demonstrates ViT for agricultural abnormal-image classification",
        "Does not incorporate environmental factors or relationships among "
        "farms and fields.",
        "The fusion module combines vision with engineered climate features and "
        "farm metadata before any prediction is made.",
        "ablation", "models/fusion.py",
    ),
    Paper(
        9, "AI Based Hybrid CNN-LSTM Model for Crop Disease Prediction", "—",
        "CNN + LSTM",
        "Combines spatial and temporal deep-learning features for prediction",
        "No transformer visual features, no graph learning and no "
        "physics-informed environmental knowledge.",
        "All three are present: a transformer backbone, a GNN over farms, and a "
        "physics loss carrying agronomic constraints.",
        "ablation", "models/full_model.py",
    ),
    Paper(
        10, "Physics-Informed Data-Driven Model for Short-Term Precipitation "
            "Prediction", "—",
        "Physics-informed data-driven learning",
        "Integrates physical knowledge with data-driven precipitation modelling",
        "Targets weather prediction rather than crop disease; no plant imagery "
        "and no agricultural graph structure.",
        "The same physics-informed philosophy is applied to disease risk, using "
        "pathogen cardinal temperatures and moisture requirements.",
        "epidemiology", "physics/epidemiology.py",
    ),
    Paper(
        11, "Intracity Temperature Estimation by Physics-Informed Neural Network "
            "Using Meteorology and Multispectral Satellite Imagery", "—",
        "PINN + meteorological forcing + multispectral imagery",
        "Integrates physics, meteorology and imagery in one model",
        "Estimates temperature, not crop disease risk or spread across "
        "agricultural regions.",
        "Our target is agronomic disease pressure at each farm, mapped across a "
        "whole production region.",
        "risk_map", "serve/inference.py",
    ),
    Paper(
        12, "Precipitation Nowcasting with Graph Neural Network and Gated "
            "Recurrent Units", "—",
        "GNN + GRU",
        "Models spatial relationships between weather stations for nowcasting",
        "The graph is over weather stations, not farms; does not combine plant "
        "image features with disease-spread modelling.",
        "Our graph nodes are farm observations whose features fuse the leaf "
        "image with that farm's weather.",
        "graph_structure", "train/prepare.py",
    ),
    Paper(
        13, "Applying Graph Neural Networks to Predict Fungal Disease "
            "Occurrences in Precision Agriculture",
        "Samson, Lord, Carisse & Makarenkov (2026)",
        "GCN on plant-level spatial graphs with weather-derived node features",
        "GNNs beat classical baselines and stay robust to missing field data",
        "Spatial-proximity graphs within single fields only; no ViT features, "
        "no physics constraints and no multi-day forecast horizon.",
        "Our graph spans 37 districts across 15 states, node features include "
        "transformer embeddings, and the forecast runs to 7 days.",
        "multi_horizon", "graph/build.py",
    ),
    Paper(
        14, "PlantPlotGAN: A Physics-Informed Generative Adversarial Network for "
            "Plant Disease Prediction", "Lopes, Sagan & Esposito (2023)",
        "Physics-informed GAN generating synthetic multispectral plot imagery",
        "Improves early prediction by augmenting limited UAV datasets",
        "Limited to single-plot synthetic imagery; no inter-field graph "
        "modelling of spread and no climate-adaptive forecasting.",
        "Physics enters both as a simulated inter-field epidemic and as a "
        "training constraint, rather than as a generator of synthetic plots.",
        "contagion", "physics/contagion.py",
    ),
    Paper(
        15, "MMST-ViT: Climate Change-Aware Crop Yield Prediction via Multi-Modal "
            "Spatial-Temporal Vision Transformer",
        "Lin, Crawford, Guillot, Zhang et al. (2023)",
        "Multi-modal ViT fusing satellite imagery with meteorological data",
        "First ViT jointly capturing seasonal weather and long-term climate for "
        "county-level yield forecasting",
        "Targets crop yield, not disease; lacks graph-based modelling of "
        "transmission between farms and physics-informed constraints.",
        "Same multi-modal spirit, but the target is disease risk and the "
        "spatial structure is an explicit graph rather than a grid.",
        "multi_horizon", "models/full_model.py",
    ),
    Paper(
        16, "Resource-Efficient Few-Shot Plant Disease Classification via "
            "Quantized Low-Rank Adapters in Vision Transformers",
        "Bayat Toksoz & Isik (2026)",
        "DINOv2 self-supervised ViT backbone with QLoRA-tuned Prototypical Network",
        "Accurate few-shot classification while tuning under 1% of parameters",
        "Purely image-based classification; no climate variables, no inter-farm "
        "graph and no future disease-risk forecasting.",
        "We also use DINOv2 and also freeze it, but feed its embeddings into a "
        "climate-aware graph forecaster.",
        "backbone_benchmark", "models/backbones.py",
    ),
    Paper(
        17, "Stimator: A Method in Agriculture CPS Framework to Estimate Severity "
            "of Plant Diseases using Graph Neural Network",
        "Kethineni, Mohanty & Kougianos (2023)",
        "GNN over a farmland graph of diseased locations",
        "Introduces a combined severity metric of affected and spread area",
        "Graph nodes encode disease locations only - no visual transformer "
        "features, no weather variables and no future-horizon forecasting.",
        "Every node carries a fused vector of transformer embedding, engineered "
        "climate and farm metadata.",
        "node_features", "models/fusion.py",
    ),
    Paper(
        18, "Spatio-Temporal Prediction of Crop Disease Severity for Agricultural "
            "Emergency Management Based on Recurrent Neural Networks",
        "Xu, Wang & Chen (2018)",
        "Spatio-temporal RNN with ensemble learning on remote-sensing and "
        "bioclimatic data",
        "Models spatial and temporal dependence in wheat yellow-rust severity",
        "Relies on RNNs rather than transformer or graph architectures; no ViT "
        "features, no physics-informed learning, no explicit inter-field graph.",
        "Transformer for vision, GNN for space, GRU only inside the horizon "
        "decoder, plus an explicit physics loss.",
        "ablation", "models/heads.py",
    ),
)


# Demonstration catalogue: what each live demo shows and which endpoint serves it.
DEMOS: dict[str, dict] = {
    "contagion": {
        "title": "Wind-borne spread between farms",
        "blurb": "The simulated epidemic on the real farm network, and the "
                 "measurement showing a farm's future risk depends on what its "
                 "neighbours are carrying today.",
        "kind": "contagion",
    },
    "graph_structure": {
        "title": "Inter-farm spatio-temporal graph",
        "blurb": "Farms linked by proximity in space and time, with directed "
                 "downwind edges for wind-borne spore transport.",
        "kind": "graph",
    },
    "physics_constraints": {
        "title": "Physics-informed constraints",
        "blurb": "Each agronomic constraint evaluated live on the trained model, "
                 "showing that legal behaviour costs nothing and violations are "
                 "penalised.",
        "kind": "physics",
    },
    "multi_horizon": {
        "title": "Multi-horizon disease-risk forecast",
        "blurb": "Risk predicted 1, 3, 5 and 7 days ahead for a real farm, with "
                 "skill measured against the weather that actually followed.",
        "kind": "forecast",
    },
    "backbone_benchmark": {
        "title": "Four transformer backbones compared",
        "blurb": "DINOv2, ViT-B/16, Swin-T and DeiT-S under one identical "
                 "downstream pipeline on a leaf-disjoint test set.",
        "kind": "table",
    },
    "epidemiology": {
        "title": "Pathogen environmental response",
        "blurb": "Published cardinal temperatures and moisture requirements, "
                 "evaluated on the real weather recorded at a farm.",
        "kind": "epidemiology",
    },
    "climate_niche": {
        "title": "Climate niche of each disease",
        "blurb": "Mean temperature and leaf wetness at which each of the 38 "
                 "classes was observed - the signal the climate branch learns.",
        "kind": "niche",
    },
    "ablation": {
        "title": "Component ablation study",
        "blurb": "What each part of the architecture is actually worth, measured "
                 "by removing it.",
        "kind": "table",
    },
    "risk_map": {
        "title": "Regional disease-risk map",
        "blurb": "Disease pressure at every farm growing a crop on a chosen "
                 "date, computed from real recorded weather.",
        "kind": "map",
    },
    "node_features": {
        "title": "What a graph node actually carries",
        "blurb": "The fused node vector: transformer embedding, engineered "
                 "climate features and farm metadata.",
        "kind": "node",
    },
}


def as_records() -> list[dict]:
    return [asdict(p) for p in PAPERS]


def demo_catalogue() -> dict[str, dict]:
    return DEMOS
