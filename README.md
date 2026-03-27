生成数据集：

python generator.py --api-key xxx --base-url xxx --model gpt-4o-mini --num-personas 10

测试baseline：

python test_class.py --models gpt-4o-mini --api-key xxx --base-url xxx --enable-llm-eval --eval-api-key xxx --eval-base-url xxx 

测试我们的模型：

python test_dialogue_only_finmem.py --model gpt-4o-mini --api-key xxx --base-url xxx --enable-llm-eval --eval-api-key xxx --eval-base-url xxx  #--no-memory 

--no-memory控制是否有记忆