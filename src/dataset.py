import json
import random

categories=["Software Engineer", "Data Analyst", "Product Manager", "HR Executive", "Sales Associate"]

statuses= ["Applied", "Screening", "Interview Scheduled", "Offered", "Rejected"]

def generate_dataset(seed=42, total_records=45):

    random.seed(seed)
    records=[]

    record_counter=1

    while True:
        records=[]
        random.seed(seed)

        for i in range(1, total_records + 1):
            record_id = f"REC-{i:03d}"
            category = random.choice(categories)
            status = random.choice(statuses)
            if category == "Software Engineer":
                salary = random.randint(600000, 2200000)
            elif category == "Product Manager":
                salary = random.randint(1000000, 2800000)
            elif category == "Data Analyst":
                salary = random.randint(500000, 1800000)
            elif category in ["HR Executive", "Sales Associate"]:
                salary = random.randint(350000, 1200000)
            else:
                salary = random.randint(400000, 1500000)
                
            days_since_created = random.randint(0, 30)

            flagged_priority_review = random.random() < 0.20
            
            records.append({
                "record_id": record_id,
                "category": category,
                "status": status,
                "expected_salary_inr": salary,
                "days_since_created": days_since_created,
                "flagged_priority_review": flagged_priority_review
            })

        cat_counts = {cat: sum(1 for r in records if r["category"] == cat) for cat in categories}
        status_counts = {st: sum(1 for r in records if r["status"] == st) for st in statuses}
        flagged_count = sum(1 for r in records if r["flagged_priority_review"])
        flagged_pct = (flagged_count / total_records) * 100

        if all(c >= 3 for c in cat_counts.values()) and \
            all(s >= 1 for s in status_counts.values()) and \
            (10.0 <= flagged_pct <= 30.0):
            break
        else:
            seed += 1

    print(f"Dataset generated successfully with seed {seed}!")
    print(f"Total Records: {len(records)}")
    print(f"Category Counts: {cat_counts}")
    print(f"Status Counts: {status_counts}")
    print(f"Flagged Priority Review Percentage: {flagged_pct:.2f}%")
    
    return records

if __name__ == "__main__":
    data= generate_dataset()

    with open("data/job_application.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Saved to data/job_application.json")    

