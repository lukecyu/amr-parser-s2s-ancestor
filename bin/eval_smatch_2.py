from ancestor_amr.evaluation import compute_smatch_2


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    # Multiple input parameters
    parser.add_argument(
        "test_path",
        type=str
    )
    parser.add_argument(
        "predict_path",
        type=str
    )
    args = parser.parse_args()
    
    print(compute_smatch_2(args.test_path, args.predict_path))
