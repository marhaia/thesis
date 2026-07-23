"""
Diagnostic script: Jokinen 2020 Visual Search Model
Runs element detection and search-time prediction on bmw_route.png.

This is a manual diagnostic script, NOT a pytest test. All heavy imports and all
model execution live inside ``main()`` behind an ``if __name__ == "__main__"``
guard, so pytest collection never triggers inference or writes output files.
Run manually with: python cognitive/test_jokinen.py
"""
import os
import sys


def main() -> int:
    # Resolve the repository root relative to this file so the script is portable.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import cv2
    import numpy as np
    from cognitive.element_detector import detect_elements
    from cognitive.jokinen_model import JokinenSearchModel, JokinenParams

    # Load test image
    img = cv2.imread("stage1/data/screenshots/bmw_route.png")
    print(f"Image shape: {img.shape}")

    # Step 1: Detect elements
    print("\n" + "=" * 60)
    print("STEP 1: Element Detection")
    print("=" * 60)
    elements = detect_elements(img)
    print(f"Detected {len(elements)} elements")
    for e in elements[:8]:
        print(f"  #{e['id']:2d}: bbox={e['bbox']}, "
              f"color={e['color_category']:6s}, "
              f"ang_size={e['angular_size']:.2f} deg")

    # Step 2: Run Jokinen model (feature-only, no saliency map)
    print("\n" + "=" * 60)
    print("STEP 2: Jokinen Model — Feature-Only Mode")
    print("=" * 60)
    params = JokinenParams(n_simulations=50, random_seed=42)
    model = JokinenSearchModel(params)
    results = model.predict_search_times(elements, image_shape=img.shape[:2])

    print(f"Mean search time: {results['mean_search_time_s']:.3f} s")
    print(f"Max search time:  {results['max_search_time_s']:.3f} s")
    print(f"Min search time:  {results['min_search_time_s']:.3f} s")
    print(f"Std deviation:    {results['search_time_std_s']:.3f} s")
    print(f"Difficulty:       {results['predicted_difficulty']}")
    print(f"N elements:       {results['n_elements']}")

    # Show top 5 hardest elements
    sorted_elems = sorted(results["per_element"],
                          key=lambda x: x["search_time_s"], reverse=True)
    print("\nHardest to find (top 5):")
    for e in sorted_elems[:5]:
        print(f"  #{e['id']:2d}: {e['search_time_s']:.3f}s "
              f"({e['fixation_count']:.1f} fix), "
              f"color={e['color_category']}")

    print("\nEasiest to find (top 5):")
    for e in sorted_elems[-5:]:
        print(f"  #{e['id']:2d}: {e['search_time_s']:.3f}s "
              f"({e['fixation_count']:.1f} fix), "
              f"color={e['color_category']}")

    # Step 3: With UMSI++ saliency
    print("\n" + "=" * 60)
    print("STEP 3: Jokinen Model — With UMSI++ Saliency")
    print("=" * 60)
    saliency_ok = False
    try:
        from saliency.umsi_model import UMSIPlus

        WEIGHTS = "saliency/weights/model_weights/saliency_models/UMSI++/umsi++.hdf5"
        IMAGE_PATH = "stage1/data/screenshots/bmw_route.png"

        print("Loading UMSI++ model...")
        umsi = UMSIPlus(weights_path=WEIGHTS)

        print("Predicting saliency map...")
        sal_map, aux_classif = umsi.predict_saliency(IMAGE_PATH, return_classif=True)
        print(f"Saliency map shape: {sal_map.shape}, "
              f"range: [{sal_map.min():.4f}, {sal_map.max():.4f}]")
        # aux_classif is the RAW auxiliary output head. It is UNVALIDATED and is
        # NOT a semantic UI-type classifier (zero training-loss weight upstream);
        # print only the raw vector, never a derived class label.
        print(f"aux_classif (RAW, UNVALIDATED, non-semantic): "
              f"{np.asarray(aux_classif).ravel().tolist()}")

        # Run Jokinen with saliency
        results_sal = model.predict_search_times(
            elements,
            saliency_map=sal_map,
            image_shape=img.shape[:2]
        )

        print(f"\nWith UMSI++ saliency:")
        print(f"  Mean search time: {results_sal['mean_search_time_s']:.3f} s")
        print(f"  Max search time:  {results_sal['max_search_time_s']:.3f} s")
        print(f"  Min search time:  {results_sal['min_search_time_s']:.3f} s")
        print(f"  Difficulty:       {results_sal['predicted_difficulty']}")

        # Compare
        delta = results_sal['mean_search_time_s'] - results['mean_search_time_s']
        print(f"\n  Delta (saliency - feature-only): {delta:+.3f} s")

        # Show how saliency changes rankings
        sorted_sal = sorted(results_sal["per_element"],
                            key=lambda x: x["search_time_s"], reverse=True)
        print("\n  Hardest with saliency (top 5):")
        for e in sorted_sal[:5]:
            print(f"    #{e['id']:2d}: {e['search_time_s']:.3f}s "
                  f"({e['fixation_count']:.1f} fix)")
        saliency_ok = True

    except ImportError as ex:
        print(f"  [SKIP] UMSI++ not available: {ex}")
    except Exception as ex:
        import traceback
        print(f"  [ERROR] {ex}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    if saliency_ok:
        print("Jokinen 2020 feature-only + UMSI++ saliency stages completed.")
        return 0
    print("Jokinen 2020 feature-only stage completed; UMSI++ saliency stage "
          "did NOT complete (see SKIP/ERROR above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
