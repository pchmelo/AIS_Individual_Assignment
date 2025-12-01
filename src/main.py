from pipeline import DatasetEvaluationPipeline

def main():
    print("Dataset Quality and Fairness Evaluation System")
    print("="*80)
    
    use_api = 2     # 1=IBM Granite, 2=Grok (API)
    user_prompt = "audit adult-all dataset"
    
    if not user_prompt:
        print("No input provided. Exiting.")
        return
    
    print(f"\nInitializing pipeline...")
    print(f"Model: {'Grok (API)' if use_api else 'IBM Granite (Local)'}")
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
     