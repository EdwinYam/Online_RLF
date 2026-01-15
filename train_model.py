# =============================================================================
# 原本的模型載入/訓練流程 (略去細節，只保留與 ODL 相關部分)
# =============================================================================
sampling_method = "v1"
v3_mode = "rlf_keep"
v3_window_size = 60
v3_step = 50
v3_rlf_window_step = [30]

csv_list = [
    "/datastore/AI_RLF/training_data_highway.csv",
    "/datastore/AI_RLF/training_data_subway.csv",
    "/datastore/AI_RLF/training_data_hst.csv",
    "/datastore/AI_RLF/training_data_hst_1.csv",
]
dataset_names = [os.path.splitext(os.path.basename(p))[0] for p in csv_list]
results_dir = "results_by_dataset"
os.makedirs(results_dir, exist_ok=True)

best_model_path = os.path.join(results_dir, "best_model.pth")
features_path = os.path.join(results_dir, "final_features.json")
os.makedirs(results_dir, exist_ok=True)

num_classes = 4
all_labels = [0, 1, 2, 3]

model = None
final_features = None

if os.path.exists(best_model_path) and os.path.exists(features_path):
    print(f"載入已存在模型: {best_model_path}")
    with open(features_path, "r") as f:
        final_features = json.load(f)
    sample_df = pd.read_csv(csv_list[0])
    sample_df["rlf_reason"] = sample_df["rlf_reason"].replace([" ", np.nan], 3)
    sample_df["rlf_reason"] = sample_df["rlf_reason"].astype(int)
    X_sample = sample_df[final_features].fillna(0).replace(
        [np.inf, -np.inf], 0
    )
    input_size = X_sample.shape[1]
    model = NeuralNet(input_size, num_classes).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
else:
    print("未找到已存在模型，開始訓練新模型...")

    train_X_list, train_y_list = [], []
    val_X_list, val_y_list = [], []
    test_X_dict, test_y_dict = {}, {}
    test_df_dict = {}
    pci_filtered_test_df_dict = {}
    pci_filter_keep_indices_dict = {}
    pci_filtered_test_df_trainonly_dict = {}
    pci_filter_keep_indices_trainonly_dict = {}
    pci_filtered_test_df_testonly_dict = {}
    pci_filter_keep_indices_testonly_dict = {}
    final_features = None

    for csv_path, dataset_name in zip(csv_list, dataset_names):
        print(f"\n=== 處理 {csv_path} ===")
        df = pd.read_csv(csv_path)
        df["rlf_reason"] = df["rlf_reason"].replace([" ", np.nan], 3)
        df["rlf_reason"] = df["rlf_reason"].astype(int)
        excluded_features = [
            "frc_64us_10",
            "frc_64us_1",
            "file_source_10",
            "file_source_1",
            "pci_1",
            "pci_2",
            "pci_3",
            "pci_4",
            "pci_5",
            "pci_6",
            "pci_7",
            "pci_8",
            "pci_9",
            "pci_10",
        ]
        X = df.drop(["rlf_reason"] + excluded_features, axis=1)
        ho_cols = [col for col in X.columns if "_ho_" in col]
        if ho_cols:
            print(f"{dataset_name} 移除含有 _ho_ 的欄位: {ho_cols}")
            X = X.drop(columns=ho_cols)
        y = df["rlf_reason"].values
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        y = np.nan_to_num(y, nan=3, posinf=3, neginf=3)
        constant_features = [col for col in X.columns if X[col].nunique() == 1]
        X = X.drop(columns=constant_features)
        if final_features is None:
            k = X.shape[1]
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X, y)
            selected_feature_indices = selector.get_support(indices=True)
            selected_features = X.columns[selected_feature_indices].tolist()
            keywords = [
                "total_ack",
                "_rb",
                "duplex",
                "is_static",
                "ack_ratio",
                "bw",
                "freq",
                "dl_mcs_idx",
                "pci",
            ]
            filtered_features = [
                feat
                for feat in selected_features
                if not any(kw in feat for kw in keywords)
            ]
            final_features = filtered_features
            with open(features_path, "w") as f:
                json.dump(final_features, f, indent=2)
            
            new_lst = [remove_dot_zero(s) for s in final_features]
            with open("my_list.json", "w") as f:
                json.dump(new_lst, f)

        X = df[final_features].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        y = df["rlf_reason"].values
        y = np.nan_to_num(y, nan=3, posinf=3, neginf=3)
        print(f"{dataset_name} 欄位數: {X.shape[1]}")
        N = len(X)
        train_end = int(N * 0.7)
        val_end = int(N * 0.85)
        X_train = X.iloc[:train_end]
        y_train = y[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y[val_end:]
        test_df = df.iloc[val_end:].reset_index(drop=True)
        test_df_dict[dataset_name] = test_df

        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test_full = df.iloc[val_end:].reset_index(drop=True)
        pci_1_values = test_df["pci_1"].unique()
        pci_1_to_remove_all = []
        for v in pci_1_values:
            mask_train = (df_train["pci_1"] == v) & (
                df_train["rlf_reason"] != 3
            )
            mask_val = (df_val["pci_1"] == v) & (df_val["rlf_reason"] != 3)
            mask_test = (df_test_full["pci_1"] == v) & (
                df_test_full["rlf_reason"] != 3
            )
            if not (mask_train.any() or mask_val.any() or mask_test.any()):
                pci_1_to_remove_all.append(v)
        mask_keep_all = ~test_df["pci_1"].isin(pci_1_to_remove_all)
        pci_filtered_test_df_all = test_df[mask_keep_all].reset_index(
            drop=True
        )
        pci_filtered_test_df_dict[dataset_name] = pci_filtered_test_df_all
        pci_filter_keep_indices_dict[dataset_name] = test_df[
            mask_keep_all
        ].index
        out_dir = os.path.join(results_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        pci_filtered_test_df_all.to_csv(
            os.path.join(out_dir, "pci_filtered_test_data.csv"), index=False
        )
        print(
            f"[{dataset_name}] pci_1 filtered test data (train+val+test) saved, {len(test_df)} -> {len(pci_filtered_test_df_all)} rows"
        )

        pci_1_to_remove_trainonly = []
        for v in pci_1_values:
            mask_train = (df_train["pci_1"] == v) & (
                df_train["rlf_reason"] != 3
            )
            if not mask_train.any():
                pci_1_to_remove_trainonly.append(v)
        mask_keep_trainonly = ~test_df["pci_1"].isin(pci_1_to_remove_trainonly)
        pci_filtered_test_df_trainonly = test_df[
            mask_keep_trainonly
        ].reset_index(drop=True)
        pci_filtered_test_df_trainonly_dict[dataset_name] = (
            pci_filtered_test_df_trainonly
        )
        pci_filter_keep_indices_trainonly_dict[dataset_name] = test_df[
            mask_keep_trainonly
        ].index
        pci_filtered_test_df_trainonly.to_csv(
            os.path.join(out_dir, "pci_filtered_test_data_trainonly.csv"),
            index=False,
        )
        print(
            f"[{dataset_name}] pci_1 filtered test data (train only) saved, {len(test_df)} -> {len(pci_filtered_test_df_trainonly)} rows"
        )

        pci_1_to_remove_testonly = []
        for v in pci_1_values:
            mask_test = (test_df["pci_1"] == v) & (
                test_df["rlf_reason"] != 3
            )
            if not mask_test.any():
                pci_1_to_remove_testonly.append(v)
        mask_keep_testonly = ~test_df["pci_1"].isin(pci_1_to_remove_testonly)
        pci_filtered_test_df_testonly = test_df[
            mask_keep_testonly
        ].reset_index(drop=True)
        pci_filtered_test_df_testonly_dict[dataset_name] = (
            pci_filtered_test_df_testonly
        )
        pci_filter_keep_indices_testonly_dict[dataset_name] = test_df[
            mask_keep_testonly
        ].index
        pci_filtered_test_df_testonly.to_csv(
            os.path.join(out_dir, "pci_filtered_test_data_testonly.csv"),
            index=False,
        )
        print(
            f"[{dataset_name}] pci_1 filtered test data (test only) saved, {len(test_df)} -> {len(pci_filtered_test_df_testonly)} rows"
        )

        if sampling_method == "v1":
            X_train, y_train = sample_class3_v1(X_train, y_train, step=50)
            X_val, y_val = sample_class3_v1(X_val, y_val, step=50)
        elif sampling_method == "v3":
            X_train, y_train = sample_class3_v3(
                X_train,
                y_train,
                window_size=v3_window_size,
                step=v3_step,
                rlf_window_step=v3_rlf_window_step,
                mode=v3_mode,
            )
            X_val, y_val = sample_class3_v3(
                X_val,
                y_val,
                window_size=v3_window_size,
                step=v3_step,
                rlf_window_step=v3_rlf_window_step,
                mode=v3_mode,
            )
        else:
            raise ValueError("Unknown sampling_method")
        train_X_list.append(X_train)
        train_y_list.append(y_train)
        val_X_list.append(X_val)
        val_y_list.append(y_val)
        test_X_dict[dataset_name] = X_test
        test_y_dict[dataset_name] = y_test

    def drop_duplicate_columns(df):
        return df.loc[:, ~df.columns.duplicated()]

    train_X_list = [drop_duplicate_columns(df) for df in train_X_list]
    val_X_list = [drop_duplicate_columns(df) for df in val_X_list]

    X_train_all = pd.concat(
        train_X_list + val_X_list, axis=0
    ).reset_index(drop=True)
    y_train_all = np.concatenate(train_y_list + val_y_list, axis=0)

    X_train_all = X_train_all.fillna(0).replace([np.inf, -np.inf], 0)
    y_train_all = np.nan_to_num(y_train_all, nan=3, posinf=3, neginf=3)

    X_train_tensor = torch.tensor(
        X_train_all.values, dtype=torch.float32
    ).to(device)
    y_train_tensor = torch.tensor(y_train_all, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = NeuralNet(X_train_tensor.shape[1], num_classes).to(device)
    class_weights = torch.tensor([1, 1, 1, 1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 200
    patience = 5
    best_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        print(f"[ALL] Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("總模型訓練完成。")

    cm_bin_total_dict = {
        "pci1filtered": np.zeros((2, 2), dtype=int),
        "pci1filtered_trainonly": np.zeros((2, 2), dtype=int),
        "pci1filtered_testonly": np.zeros((2, 2), dtype=int),
        "pci1filtered_seq2": np.zeros((2, 2), dtype=int),
        "pci1filtered_trainonly_seq2": np.zeros((2, 2), dtype=int),
        "pci1filtered_testonly_seq2": np.zeros((2, 2), dtype=int),
    }

    for csv_path, dataset_name in zip(csv_list, dataset_names):
        out_dir = os.path.join(results_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)

        X_test = test_X_dict[dataset_name].fillna(0).replace(
            [np.inf, -np.inf], 0
        )
        y_test = np.nan_to_num(
            test_y_dict[dataset_name], nan=3, posinf=3, neginf=3
        )
        X_test_tensor = torch.tensor(
            X_test.values, dtype=torch.float32
        ).to(device)
        model.eval()
        with torch.no_grad():
            y_test_pred_proba = (
                torch.softmax(model(X_test_tensor), dim=1).cpu().numpy()
            )
        margins = [0, 0, 0]

        def custom_predict_margin_multi_eval(proba, margins):
            preds = []
            for p in proba:
                max_idx = np.argmax(p[:3])
                if p[max_idx] > p[3] + margins[max_idx]:
                    preds.append(max_idx)
                else:
                    preds.append(3)
            return np.array(preds)

        y_test_pred = custom_predict_margin_multi_eval(
            y_test_pred_proba, margins
        )

        cm_test = confusion_matrix(y_test, y_test_pred, labels=all_labels)
        plot_confusion_matrix(
            cm_test,
            all_labels,
            f"Confusion Matrix (Test Data) - {dataset_name}",
            os.path.join(out_dir, "confusion_matrix_test.png"),
        )

        y_test_bin = (y_test != 3).astype(int)
        y_test_pred_bin = (y_test_pred != 3).astype(int)
        cm_bin = confusion_matrix(y_test_bin, y_test_pred_bin, labels=[0, 1])
        plot_confusion_matrix(
            cm_bin,
            ["RLF", "Not RLF"],
            f"Confusion Matrix (Test Data, 2-class) - {dataset_name}",
            os.path.join(out_dir, "confusion_matrix_test_2class.png"),
        )

        plt.figure(figsize=(10, 8))
        for i, class_label in enumerate(all_labels):
            try:
                precision, recall, _ = precision_recall_curve(
                    y_test == class_label, y_test_pred_proba[:, i]
                )
                plt.plot(recall, precision, label=f"Class {class_label}")
            except Exception as e:
                print(f"PR曲線 class {class_label} 發生錯誤: {e}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve (Test Data) - {dataset_name}")
        plt.legend()
        plt.savefig(os.path.join(out_dir, "pr_curve_test.png"))
        plt.close()

        with open(
            os.path.join(out_dir, "classification_report.txt"), "w"
        ) as f:
            f.write("測試數據分類報告:\n")
            f.write(
                classification_report(
                    y_test,
                    y_test_pred,
                    labels=all_labels,
                    target_names=[str(l) for l in all_labels],
                    zero_division=0,
                )
            )
        pd.DataFrame({"y_test": y_test, "y_test_pred": y_test_pred}).to_csv(
            os.path.join(out_dir, "test_pred_result.csv"), index=False
        )
        print(f"=== {dataset_name} 測試分析完成，結果已存到 {out_dir} ===")

        filter_types = [
            (
                "pci_1 filtered (train+val+test rule)",
                pci_filtered_test_df_dict,
                pci_filter_keep_indices_dict,
                "pci1filtered",
            ),
            (
                "pci_1 filtered (train only rule)",
                pci_filtered_test_df_trainonly_dict,
                pci_filter_keep_indices_trainonly_dict,
                "pci1filtered_trainonly",
            ),
            (
                "pci_1 filtered (test only rule)",
                pci_filtered_test_df_testonly_dict,
                pci_filter_keep_indices_testonly_dict,
                "pci1filtered_testonly",
            ),
        ]
        for (
            filter_name,
            filter_df_dict,
            filter_idx_dict,
            filter_tag,
        ) in filter_types:
            keep_indices = filter_idx_dict[dataset_name]
            y_test_pci_filtered = y_test[keep_indices]
            y_test_pred_pci_filtered = y_test_pred[keep_indices]
            y_test_bin_pci_filtered = (y_test_pci_filtered != 3).astype(int)
            y_test_pred_bin_pci_filtered = (
                y_test_pred_pci_filtered != 3
            ).astype(int)
            cm_bin_pci_filtered = confusion_matrix(
                y_test_bin_pci_filtered,
                y_test_pred_bin_pci_filtered,
                labels=[0, 1],
            )
            plot_confusion_matrix(
                cm_bin_pci_filtered,
                ["RLF", "Not RLF"],
                f"Confusion Matrix (Test Data, 2-class, {filter_name}) - {dataset_name}",
                os.path.join(
                    out_dir, f"confusion_matrix_test_2class_{filter_tag}.png"
                ),
            )
            print(
                f"[{dataset_name}] {filter_name} 2x2混淆矩陣:\n{cm_bin_pci_filtered}"
            )
            cm_bin_total_dict[filter_tag] += cm_bin_pci_filtered

            y_test_pred_bin_pci_filtered_seq2 = apply_seq2_rule_r0n1(
                y_test_pred_pci_filtered
            )
            cm_bin_pci_filtered_seq2 = confusion_matrix(
                y_test_bin_pci_filtered,
                y_test_pred_bin_pci_filtered_seq2,
                labels=[0, 1],
            )
            plot_confusion_matrix(
                cm_bin_pci_filtered_seq2,
                ["RLF", "Not RLF"],
                f"Confusion Matrix (Test Data, 2-class, {filter_name} + seq2 rule) - {dataset_name}",
                os.path.join(
                    out_dir,
                    f"confusion_matrix_test_2class_{filter_tag}_seq2.png",
                ),
            )
            print(
                f"[{dataset_name}] {filter_name} + seq2 rule 2x2混淆矩陣:\n{cm_bin_pci_filtered_seq2}"
            )
            cm_bin_total_dict[filter_tag + "_seq2"] += cm_bin_pci_filtered_seq2

    for filter_tag, cm_bin_total in cm_bin_total_dict.items():
        plot_confusion_matrix(
            cm_bin_total,
            ["RLF", "Not RLF"],
            f"Confusion Matrix (ALL Test Data, 2-class, {filter_tag})",
            os.path.join(
                results_dir, f"confusion_matrix_ALL_2class_{filter_tag}.png"
            ),
        )
        print(f"ALL {filter_tag} 2x2混淆矩陣:\n{cm_bin_total}")

    print("全部資料集分析完成。")

print("使用模型進行後續 ODL/PCA 推論流程。")