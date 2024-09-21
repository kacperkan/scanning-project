#!/bin/bash
DATASET_PATH="/home/kkania/Datasets/scanning/sdfstudio/rf-00/"
methods=(
    neus-facto
    #bakedsdf
    #neus-facto-angelo
    #neus
    #neus-acc
    # neuralangelo
)

train () {
    # Run experiments
    for method in "${methods[@]}"
    do
        case ${method} in
            neus-facto)
                addiotnal_args="
                --pipeline.model.sdf-field.use-n-dot-v True \
                --pipeline.model.sdf-field.hash-features-per-level 4 \
                --pipeline.model.sdf-field.log2-hashmap-size 23
                "
                ;;
            neus-facto-angelo)
                addiotnal_args="
                --pipeline.model.sdf-field.log2-hashmap-size 21 \
                --pipeline.model.sdf-field.hash-features-per-level 4
                "
                ;;
            neus)
                addiotnal_args="
                --pipeline.model.sdf-field.num-layers 8 \
                --pipeline.model.sdf-field.hidden-dim 128 \
                --pipeline.model.sdf-field.geo-feat-dim 128 \
                --pipeline.model.sdf-field.hidden-dim-color 128
                "
                ;;
            neus-acc)
                addiotnal_args="
                --pipeline.model.sdf-field.num-layers 8 \
                --pipeline.model.sdf-field.hidden-dim 128 \
                --pipeline.model.sdf-field.geo-feat-dim 128 \
                --pipeline.model.sdf-field.hidden-dim-color 128
                "
                ;;
            neuralangelo)
                addiotnal_args="
                --pipeline.model.sdf-field.num-layers-color 2 \
                --pipeline.model.sdf-field.log2-hashmap-size 21 \
                --pipeline.model.sdf-field.hash-features-per-level 4 \
                --pipeline.datamanager.train-num-rays-per-batch 512
                "
                ;;
            *)
                addiotnal_args="--pipeline.datamanager.train-num-rays-per-batch 2048"
                ;;
        esac

        echo "Running experiment for method: ${method}"
        ns-train ${method} \
            --pipeline.model.sdf-field.inside-outside False \
            --pipeline.model.fg-mask-loss-mult 0.1 \
            --pipeline.model.background-model mlp  \
            --vis tensorboard \
            --experiment-name "" \
            --timestamp "" \
            ${addiotnal_args} \
            --output_dir outputs/rf-00-additional/ \
            --trainer.max-num-iterations 100001 \
            --trainer.steps-per-save 10000 \
            sdfstudio-data --data ${DATASET_PATH} \
            --include-foreground-mask True \
            --train_val_no_overlap True \
            --skip_every_for_val_split 4
    done
}

extract () {
    for method in "${methods[@]}"
    do
        echo "Extracting mesh for method: ${method}"
        ns-extract-mesh \
            --load-config outputs/rf-00-additional/${method}/config.yml \
            --output-path meshes/rf-00-additional/${method}/mesh.ply \
            --bounding-box-min -2.0 -2.0 -2.0 \
            --bounding-box-max 2.0 2.0 2.0 \
            --resolution 2048 \
            --marching_cube_threshold 0.0 \
            --create_visibility_mask False
    done
}

visualize () {
    # local_methods=("agisoft" "${methods[@]}")
    local_methods=("${methods[@]}")
    for method in "${local_methods[@]}"
    do
        ns-render-mesh \
            --meshfile meshes/rf-00-additional/${method}/mesh.ply \
            --traj ellipse \
            --fps 30 \
            --num_views 240 \
            --output_path renders/rf-00-additional/${method}/render.mp4 \
            --reverse-cameras \
            sdfstudio-data \
            --data ${DATASET_PATH}
    done
}

visualize_rgb () {
    # local_methods=("agisoft" "${methods[@]}")
    local_methods=("${methods[@]}")
    for method in "${local_methods[@]}"
    do
        ns-render \
            --load-config outputs/rf-00-additional/${method}/config.yml \
            --traj spiral \
            --output-path renders_rgb/rf-00-additional/${method}/render.mp4
    done
}

train
extract
visualize
visualize_rgb


