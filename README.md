while true; do
  if git diff --quiet model_data.pkl; then
    sleep 10  # Wait 10 seconds
  else
    git add model_data.pkl
    git commit -m "Auto-updated model_data.pkl"
    git push origin main
  fi
done
 



paste this in terminal first and then create new terminal and work on it
