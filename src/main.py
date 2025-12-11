from pipeline import DatasetEvaluationPipeline

def main():
    print("Dataset Quality and Fairness Evaluation System")
    print("="*80)
    
    # Model selection: 0=IBM Granite (Local), 1=Grok (API), 2=Gemini (API)
    use_api = 2
    user_prompt = "audit adult-all dataset"
    
    if not user_prompt:
        print("No input provided. Exiting.")
        return
    
    model_names = {
        0: "IBM Granite (Local)",
        1: "Grok (API)",
        2: "Google Gemini (API)"
    }
    
    print(f"\nInitializing pipeline...")
    print(f"Model: {model_names.get(use_api, 'Unknown')}")
    print(f"User prompt: {user_prompt}")
    
    try:
        pipeline = DatasetEvaluationPipeline(use_api_model=use_api)
        results = pipeline.evaluate_dataset(user_prompt)
        pipeline.generate_report()
        print("\nEvaluation completed successfully.")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
     